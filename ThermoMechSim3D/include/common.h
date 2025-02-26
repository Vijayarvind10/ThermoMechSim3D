#ifndef COMMON_H
#define COMMON_H

#include <vector>
#include <string>
#include <iostream>
#include <cuda_runtime.h>

// Error checking macro for CUDA calls
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << " line " << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Simulation configuration
struct SimulationConfig {
    // Domain dimensions
    int dim_x, dim_y, dim_z;
    float dx, dy, dz;
    
    // Time stepping
    float dt;
    int num_steps;
    int output_interval;
    
    // Physical parameters
    float ref_temp;
    float ambient_temp;
    float stress_threshold;
    
    // IO parameters
    std::string output_dir;
    std::string power_map_file;
    std::string material_db_file;
    std::string package_template;
    
    // Convergence criteria
    float convergence_threshold;
    int max_iterations;
    
    // GPU configuration
    int device_id;
    bool use_async_transfers;
    int num_streams;
};

// Material properties
struct Material {
    std::string name;
    std::string category;  // semiconductor, metal, dielectric, etc.
    
    // Thermal properties
    float thermal_conductivity;  // W/(m·K)
    float specific_heat;         // J/(kg·K)
    float density;               // kg/m³
    
    // Mechanical properties
    float youngs_modulus;        // Pa
    float poissons_ratio;        // dimensionless
    float thermal_expansion;     // 1/K
    float yield_strength;        // Pa
    float ultimate_strength;     // Pa
    float melting_point;         // K
    float fracture_toughness;    // Pa·m^(1/2)
    
    // Electrical properties (if needed)
    float electrical_resistivity; // Ω·m
    
    // Reliability data
    float fatigue_coefficient;
    float fatigue_exponent;
    
    // Reference
    std::string source;          // Reference for the data
};

// Layer in 3D-IC stackup
struct Layer {
    std::string material_name;
    float thickness;
    std::string power_map_file;
    bool thermal_interface;
    std::vector<float> custom_props;
};

// 3D-IC package configuration
struct PackageConfig {
    std::string name;
    std::vector<Layer> stackup;
    std::string boundary_top;
    std::string boundary_bottom;
    std::string boundary_sides;
    float ambient_temp;
    float convection_coefficient;
};

#endif // COMMON_H 