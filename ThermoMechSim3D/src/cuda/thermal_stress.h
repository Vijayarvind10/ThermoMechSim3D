#ifndef THERMAL_STRESS_H
#define THERMAL_STRESS_H

#include <cuda_runtime.h>

// Constants for material property indexing
#define MATERIAL_PROPS_SIZE 10
#define PROP_CONDUCTIVITY 0
#define PROP_DENSITY 1
#define PROP_SPECIFIC_HEAT 2
#define PROP_YOUNGS_MODULUS 3
#define PROP_POISSONS_RATIO 4
#define PROP_THERMAL_EXPANSION 5
#define PROP_YIELD_STRENGTH 6
#define PROP_ULTIMATE_STRENGTH 7
#define PROP_MELTING_POINT 8
#define PROP_FRACTURE_TOUGHNESS 9

#define MAX_CRITICAL_POINTS 10000

// Structure to hold material information for each grid point
struct MaterialPoint {
    int material_id;
    float properties[4];  // Additional point-specific properties if needed
};

// Main CUDA kernel for thermal-mechanical stress simulation
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
);

// Kernel to check for critical stress regions
__global__ void check_critical_stress(
    const float* stress_field,
    int* critical_points,
    int* critical_count,
    float threshold,
    int dim_x, int dim_y, int dim_z
);

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
    cudaStream_t stream = 0
);

// Host function to find critical stress regions
int findCriticalStressRegions(
    const float* d_stress_field,
    int* d_critical_points,
    float threshold,
    int dim_x, int dim_y, int dim_z,
    cudaStream_t stream = 0
);

#endif // THERMAL_STRESS_H 