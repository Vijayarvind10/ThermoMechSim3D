#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime.h>
#include "thermal_stress.h"

// Boundary condition types
enum BoundaryConditionType {
    BC_CONSTANT_TEMP,
    BC_CONVECTION,
    BC_ADIABATIC
};

// Initialize temperature field kernel
__global__ void initializeTemperatureKernel(float* temperature, float ambient_temp, 
                                           int dim_x, int dim_y, int dim_z);

// Apply boundary conditions kernel
__global__ void applyBoundaryConditionsKernel(float* temperature, const MaterialPoint* material_grid,
                                             float ambient_temp, float convection_coeff,
                                             int dim_x, int dim_y, int dim_z,
                                             BoundaryConditionType top_bc,
                                             BoundaryConditionType bottom_bc,
                                             BoundaryConditionType sides_bc,
                                             float top_bc_value,
                                             float bottom_bc_value);

// Host function to initialize temperature field
void initializeTemperatureField(float* d_temperature, float ambient_temp,
                               int dim_x, int dim_y, int dim_z,
                               cudaStream_t stream = 0);

// Host function to apply boundary conditions
void applyBoundaryConditions(float* d_temperature, const MaterialPoint* d_material_grid,
                            float ambient_temp, float convection_coeff,
                            int dim_x, int dim_y, int dim_z,
                            BoundaryConditionType top_bc,
                            BoundaryConditionType bottom_bc,
                            BoundaryConditionType sides_bc,
                            float top_bc_value, float bottom_bc_value,
                            cudaStream_t stream = 0);

// External texture reference for material properties (defined in thermal_stress.cu)
extern texture<float, 1, cudaReadModeElementType> materialTex;

#endif // CUDA_UTILS_H 