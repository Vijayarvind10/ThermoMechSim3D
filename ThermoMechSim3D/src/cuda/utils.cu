#include "utils.h"
#include <iostream>

__global__ void initializeTemperatureKernel(float* temperature, float ambient_temp, 
                                           int dim_x, int dim_y, int dim_z) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (idx >= dim_x || idy >= dim_y || idz >= dim_z) return;
    
    int global_idx = idz * dim_x * dim_y + idy * dim_x + idx;
    temperature[global_idx] = ambient_temp;
}

__global__ void applyBoundaryConditionsKernel(float* temperature, const MaterialPoint* material_grid,
                                             float ambient_temp, float convection_coeff,
                                             int dim_x, int dim_y, int dim_z,
                                             BoundaryConditionType top_bc,
                                             BoundaryConditionType bottom_bc,
                                             BoundaryConditionType sides_bc,
                                             float top_bc_value,
                                             float bottom_bc_value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Handle top boundary (z = dim_z - 1)
    if (idx < dim_x && idy < dim_y) {
        int global_idx = (dim_z - 1) * dim_x * dim_y + idy * dim_x + idx;
        
        switch (top_bc) {
            case BC_CONSTANT_TEMP:
                temperature[global_idx] = top_bc_value;
                break;
                
            case BC_CONVECTION: {
                int inner_idx = (dim_z - 2) * dim_x * dim_y + idy * dim_x + idx;
                float dx = 1.0f; // Grid spacing in z-direction (normalized)
                int mat_id = material_grid[global_idx].material_id;
                float k = tex1Dfetch(materialTex, mat_id * MATERIAL_PROPS_SIZE + PROP_CONDUCTIVITY);
                
                // Convection boundary: -k*dT/dz = h*(T - T_ambient)
                // Using finite difference for dT/dz
                float dTdz = (temperature[global_idx] - temperature[inner_idx]) / dx;
                float h = convection_coeff;
                
                // Update temperature at boundary
                temperature[global_idx] = (temperature[inner_idx] * k / dx + h * ambient_temp) / (k / dx + h);
                break;
            }
                
            case BC_ADIABATIC: {
                int inner_idx = (dim_z - 2) * dim_x * dim_y + idy * dim_x + idx;
                temperature[global_idx] = temperature[inner_idx]; // No heat flux
                break;
            }
                
            default:
                break;
        }
    }
    
    // Handle bottom boundary (z = 0)
    if (idx < dim_x && idy < dim_y) {
        int global_idx = idy * dim_x + idx;
        
        switch (bottom_bc) {
            case BC_CONSTANT_TEMP:
                temperature[global_idx] = bottom_bc_value;
                break;
                
            case BC_CONVECTION: {
                int inner_idx = dim_x * dim_y + idy * dim_x + idx;
                float dx = 1.0f; // Grid spacing in z-direction (normalized)
                int mat_id = material_grid[global_idx].material_id;
                float k = tex1Dfetch(materialTex, mat_id * MATERIAL_PROPS_SIZE + PROP_CONDUCTIVITY);
                
                // Convection boundary: -k*dT/dz = h*(T - T_ambient)
                // Using finite difference for dT/dz
                float dTdz = (temperature[inner_idx] - temperature[global_idx]) / dx;
                float h = convection_coeff;
                
                // Update temperature at boundary
                temperature[global_idx] = (temperature[inner_idx] * k / dx + h * ambient_temp) / (k / dx + h);
                break;
            }
                
            case BC_ADIABATIC: {
                int inner_idx = dim_x * dim_y + idy * dim_x + idx;
                temperature[global_idx] = temperature[inner_idx]; // No heat flux
                break;
            }
                
            default:
                break;
        }
    }
    
    // The side boundaries handling would be similar but more complex
    // For brevity, we'll implement a simplified version
    
    // Left boundary (x = 0)
    if (idy < dim_y && idx == 0) {
        for (int idz = 0; idz < dim_z; idz++) {
            int global_idx = idz * dim_x * dim_y + idy * dim_x;
            
            if (sides_bc == BC_ADIABATIC) {
                int inner_idx = idz * dim_x * dim_y + idy * dim_x + 1;
                temperature[global_idx] = temperature[inner_idx]; // No heat flux
            }
        }
    }
    
    // Right boundary (x = dim_x - 1)
    if (idy < dim_y && idx == dim_x - 1) {
        for (int idz = 0; idz < dim_z; idz++) {
            int global_idx = idz * dim_x * dim_y + idy * dim_x + (dim_x - 1);
            
            if (sides_bc == BC_ADIABATIC) {
                int inner_idx = idz * dim_x * dim_y + idy * dim_x + (dim_x - 2);
                temperature[global_idx] = temperature[inner_idx]; // No heat flux
            }
        }
    }
    
    // Front boundary (y = 0)
    if (idx < dim_x && idy == 0) {
        for (int idz = 0; idz < dim_z; idz++) {
            int global_idx = idz * dim_x * dim_y + idx;
            
            if (sides_bc == BC_ADIABATIC) {
                int inner_idx = idz * dim_x * dim_y + dim_x + idx;
                temperature[global_idx] = temperature[inner_idx]; // No heat flux
            }
        }
    }
    
    // Back boundary (y = dim_y - 1)
    if (idx < dim_x && idy == dim_y - 1) {
        for (int idz = 0; idz < dim_z; idz++) {
            int global_idx = idz * dim_x * dim_y + (dim_y - 1) * dim_x + idx;
            
            if (sides_bc == BC_ADIABATIC) {
                int inner_idx = idz * dim_x * dim_y + (dim_y - 2) * dim_x + idx;
                temperature[global_idx] = temperature[inner_idx]; // No heat flux
            }
        }
    }
}

void initializeTemperatureField(float* d_temperature, float ambient_temp,
                               int dim_x, int dim_y, int dim_z,
                               cudaStream_t stream) {
    dim3 threads_per_block(16, 16, 1);
    dim3 blocks_per_grid(
        (dim_x + threads_per_block.x - 1) / threads_per_block.x,
        (dim_y + threads_per_block.y - 1) / threads_per_block.y,
        dim_z
    );
    
    initializeTemperatureKernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        d_temperature, ambient_temp, dim_x, dim_y, dim_z
    );
}

void applyBoundaryConditions(float* d_temperature, const MaterialPoint* d_material_grid,
                            float ambient_temp, float convection_coeff,
                            int dim_x, int dim_y, int dim_z,
                            BoundaryConditionType top_bc,
                            BoundaryConditionType bottom_bc,
                            BoundaryConditionType sides_bc,
                            float top_bc_value, float bottom_bc_value,
                            cudaStream_t stream) {
    dim3 threads_per_block(16, 16);
    dim3 blocks_per_grid(
        (dim_x + threads_per_block.x - 1) / threads_per_block.x,
        (dim_y + threads_per_block.y - 1) / threads_per_block.y
    );
    
    applyBoundaryConditionsKernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        d_temperature, d_material_grid, ambient_temp, convection_coeff,
        dim_x, dim_y, dim_z, top_bc, bottom_bc, sides_bc,
        top_bc_value, bottom_bc_value
    );
} 