#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include "../include/common.h"
#include "thermal_stress.h"

namespace cg = cooperative_groups;

// Texture memory for material properties
texture<float, 1, cudaReadModeElementType> materialTex;

// Helper function to calculate 3D index
__device__ inline int index3D(int x, int y, int z, int dimX, int dimY) {
    return z * dimX * dimY + y * dimX + x;
}

// Helper function to load data to shared memory with halo regions
__device__ void load_to_shared_memory(
    const float* temp_field,
    float temp_tile[18][18][18],
    int idx, int idy, int idz,
    int dimX, int dimY, int dimZ
) {
    int tile_idx = threadIdx.x + 1;
    int tile_idy = threadIdx.y + 1;
    int tile_idz = threadIdx.z + 1;
    
    // Load the center value
    if (idx < dimX && idy < dimY && idz < dimZ) {
        temp_tile[tile_idz][tile_idy][tile_idx] = 
            temp_field[index3D(idx, idy, idz, dimX, dimY)];
    } else {
        temp_tile[tile_idz][tile_idy][tile_idx] = 0.0f;
    }
    
    // Load halo regions (minimal logic shown for clarity)
    if (threadIdx.x == 0) {
        int halo_idx = idx - 1;
        if (halo_idx >= 0 && idy < dimY && idz < dimZ) {
            temp_tile[tile_idz][tile_idy][0] = 
                temp_field[index3D(halo_idx, idy, idz, dimX, dimY)];
        } else {
            temp_tile[tile_idz][tile_idy][0] = temp_tile[tile_idz][tile_idy][1];
        }
    }
    
    // Note: Additional halo loading code for other boundaries would be here
    // (skipped for brevity)
    
    __syncthreads();
}

// Compute heat flux at a grid point using finite difference method
__device__ float compute_heat_flux(
    float temp_tile[18][18][18],
    int tile_idx, int tile_idy, int tile_idz,
    float dx, float dy, float dz,
    int material_id
) {
    // Get material properties from texture memory
    float conductivity = tex1Dfetch(materialTex, material_id * MATERIAL_PROPS_SIZE + PROP_CONDUCTIVITY);
    float density = tex1Dfetch(materialTex, material_id * MATERIAL_PROPS_SIZE + PROP_DENSITY);
    float specific_heat = tex1Dfetch(materialTex, material_id * MATERIAL_PROPS_SIZE + PROP_SPECIFIC_HEAT);
    
    // Calculate partial derivatives for temperature
    float d2tdx2 = (temp_tile[tile_idz][tile_idy][tile_idx+1] - 
                   2.0f * temp_tile[tile_idz][tile_idy][tile_idx] + 
                   temp_tile[tile_idz][tile_idy][tile_idx-1]) / (dx * dx);
    
    float d2tdy2 = (temp_tile[tile_idz][tile_idy+1][tile_idx] - 
                   2.0f * temp_tile[tile_idz][tile_idy][tile_idx] + 
                   temp_tile[tile_idz][tile_idy-1][tile_idx]) / (dy * dy);
    
    float d2tdz2 = (temp_tile[tile_idz+1][tile_idy][tile_idx] - 
                   2.0f * temp_tile[tile_idz][tile_idy][tile_idx] + 
                   temp_tile[tile_idz-1][tile_idy][tile_idx]) / (dz * dz);
    
    // Heat equation: ρCp(∂T/∂t) = ∇·(k∇T) + Qsource
    // For steady state, ∂T/∂t = 0, so the flux is:
    float flux = conductivity * (d2tdx2 + d2tdy2 + d2tdz2);
    
    return flux;
}

// Compute stress tensor components
__device__ void compute_stress_tensor(
    float temp_tile[18][18][18],
    int tile_idx, int tile_idy, int tile_idz,
    float dx, float dy, float dz,
    int material_id,
    float stress[6],  // [σxx, σyy, σzz, σxy, σyz, σxz]
    float ref_temp
) {
    // Get material properties from texture memory
    float youngs_modulus = tex1Dfetch(materialTex, material_id * MATERIAL_PROPS_SIZE + PROP_YOUNGS_MODULUS);
    float poissons_ratio = tex1Dfetch(materialTex, material_id * MATERIAL_PROPS_SIZE + PROP_POISSONS_RATIO);
    float thermal_expansion = tex1Dfetch(materialTex, material_id * MATERIAL_PROPS_SIZE + PROP_THERMAL_EXPANSION);
    
    // Calculate temperature difference from reference
    float temp_diff = temp_tile[tile_idz][tile_idy][tile_idx] - ref_temp;
    
    // Calculate strain components due to thermal expansion
    float thermal_strain = thermal_expansion * temp_diff;
    
    // Calculate Lamé parameters
    float lambda = youngs_modulus * poissons_ratio / ((1.0f + poissons_ratio) * (1.0f - 2.0f * poissons_ratio));
    float mu = youngs_modulus / (2.0f * (1.0f + poissons_ratio));
    
    // Calculate strain gradients (simplified for this example)
    float dudx = (temp_tile[tile_idz][tile_idy][tile_idx+1] - temp_tile[tile_idz][tile_idy][tile_idx-1]) / (2.0f * dx);
    float dvdy = (temp_tile[tile_idz][tile_idy+1][tile_idx] - temp_tile[tile_idz][tile_idy-1][tile_idx]) / (2.0f * dy);
    float dwdz = (temp_tile[tile_idz+1][tile_idy][tile_idx] - temp_tile[tile_idz-1][tile_idy][tile_idx]) / (2.0f * dz);
    
    // Calculate stress tensor components (Hooke's law with thermal expansion)
    // Normal stresses
    stress[0] = 2.0f * mu * dudx + lambda * (dudx + dvdy + dwdz) - (3.0f * lambda + 2.0f * mu) * thermal_strain; // σxx
    stress[1] = 2.0f * mu * dvdy + lambda * (dudx + dvdy + dwdz) - (3.0f * lambda + 2.0f * mu) * thermal_strain; // σyy
    stress[2] = 2.0f * mu * dwdz + lambda * (dudx + dvdy + dwdz) - (3.0f * lambda + 2.0f * mu) * thermal_strain; // σzz
    
    // Shear stresses (simplified for example)
    stress[3] = mu * (dudx + dvdy); // σxy
    stress[4] = mu * (dvdy + dwdz); // σyz
    stress[5] = mu * (dwdz + dudx); // σxz
}

// Compute von Mises stress
__device__ float von_mises(float stress[6]) {
    // von Mises yield criterion: σvm = sqrt(0.5*[(σxx-σyy)² + (σyy-σzz)² + (σzz-σxx)² + 6*(σxy² + σyz² + σxz²)])
    float term1 = (stress[0] - stress[1]) * (stress[0] - stress[1]);
    float term2 = (stress[1] - stress[2]) * (stress[1] - stress[2]);
    float term3 = (stress[2] - stress[0]) * (stress[2] - stress[0]);
    float term4 = 6.0f * (stress[3]*stress[3] + stress[4]*stress[4] + stress[5]*stress[5]);
    
    return sqrtf(0.5f * (term1 + term2 + term3 + term4));
}

// Main CUDA kernel for solving thermal-mechanical stress
__global__ void solve_thermal_stress(
    const float* temp_field,
    float* new_temp_field,
    float* stress_field,
    const MaterialPoint* material_grid,
    float dt,
    float dx, float dy, float dz,
    int dim_x, int dim_y, int dim_z,
    float ref_temp,
    float* power_map
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;
    
    // Tile configuration with halo regions
    __shared__ float temp_tile[18][18][18];  // 16x16x16 + halo
    
    // Load temperature data to shared memory
    load_to_shared_memory(temp_field, temp_tile, idx, idy, idz, dim_x, dim_y, dim_z);
    __syncthreads();
    
    // Skip threads outside domain
    if (idx >= dim_x || idy >= dim_y || idz >= dim_z) return;
    
    // Get global linear index
    int global_idx = index3D(idx, idy, idz, dim_x, dim_y);
    
    // Get material properties for current point
    int material_id = material_grid[global_idx].material_id;
    
    // Tile indices for the current thread (including halo offset)
    int tile_idx = threadIdx.x + 1;
    int tile_idy = threadIdx.y + 1;
    int tile_idz = threadIdx.z + 1;
    
    // Compute heat flux
    float flux = compute_heat_flux(temp_tile, tile_idx, tile_idy, tile_idz, 
                                  dx, dy, dz, material_id);
    
    // Add heat source if available
    if (power_map) {
        flux += power_map[global_idx];
    }
    
    // Compute new temperature (explicit time stepping)
    float density = tex1Dfetch(materialTex, material_id * MATERIAL_PROPS_SIZE + PROP_DENSITY);
    float specific_heat = tex1Dfetch(materialTex, material_id * MATERIAL_PROPS_SIZE + PROP_SPECIFIC_HEAT);
    float thermal_capacity = density * specific_heat;
    
    new_temp_field[global_idx] = temp_field[global_idx] + dt * flux / thermal_capacity;
    
    // Compute stress tensor
    float stress[6];
    compute_stress_tensor(temp_tile, tile_idx, tile_idy, tile_idz,
                         dx, dy, dz, material_id, stress, ref_temp);
    
    // Compute von Mises stress and store result
    stress_field[global_idx] = von_mises(stress);
}

// Kernel to check for critical stress regions
__global__ void check_critical_stress(
    const float* stress_field,
    int* critical_points,
    int* critical_count,
    float threshold,
    int dim_x, int dim_y, int dim_z
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (idx >= dim_x || idy >= dim_y || idz >= dim_z) return;
    
    int global_idx = index3D(idx, idy, idz, dim_x, dim_y);
    
    // Check if stress exceeds critical threshold
    if (stress_field[global_idx] > threshold) {
        // Atomically add this point to the critical points array
        int point_idx = atomicAdd(critical_count, 1);
        if (point_idx < MAX_CRITICAL_POINTS) {
            critical_points[point_idx * 4] = idx;
            critical_points[point_idx * 4 + 1] = idy;
            critical_points[point_idx * 4 + 2] = idz;
            critical_points[point_idx * 4 + 3] = __float_as_int(stress_field[global_idx]);
        }
    }
}

// Host function to launch kernel with optimal configuration
void launchThermalStressKernel(
    const float* d_temp_field,
    float* d_new_temp_field,
    float* d_stress_field,
    const MaterialPoint* d_material_grid,
    float dt, float dx, float dy, float dz,
    int dim_x, int dim_y, int dim_z,
    float ref_temp,
    float* d_power_map,
    cudaStream_t stream
) {
    // Set up block and grid dimensions
    dim3 threads_per_block(16, 16, 1);
    dim3 blocks_per_grid(
        (dim_x + threads_per_block.x - 1) / threads_per_block.x,
        (dim_y + threads_per_block.y - 1) / threads_per_block.y,
        dim_z
    );
    
    // Launch the thermal-stress kernel
    solve_thermal_stress<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        d_temp_field, d_new_temp_field, d_stress_field, d_material_grid,
        dt, dx, dy, dz, dim_x, dim_y, dim_z, ref_temp, d_power_map
    );
}

// Host function to find critical stress regions
int findCriticalStressRegions(
    const float* d_stress_field,
    int* d_critical_points,
    float threshold,
    int dim_x, int dim_y, int dim_z,
    cudaStream_t stream
) {
    // Set up block and grid dimensions
    dim3 threads_per_block(16, 16, 1);
    dim3 blocks_per_grid(
        (dim_x + threads_per_block.x - 1) / threads_per_block.x,
        (dim_y + threads_per_block.y - 1) / threads_per_block.y,
        dim_z
    );
    
    // Counter for critical points
    int* d_critical_count;
    cudaMalloc(&d_critical_count, sizeof(int));
    cudaMemset(d_critical_count, 0, sizeof(int));
    
    // Launch the kernel to check for critical stress regions
    check_critical_stress<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        d_stress_field, d_critical_points, d_critical_count,
        threshold, dim_x, dim_y, dim_z
    );
    
    // Get the count back to host
    int critical_count;
    cudaMemcpy(&critical_count, d_critical_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_critical_count);
    
    return critical_count;
} 