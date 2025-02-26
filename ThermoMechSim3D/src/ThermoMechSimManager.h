#ifndef THERMO_MECH_SIM_MANAGER_H
#define THERMO_MECH_SIM_MANAGER_H

#include <string>
#include <vector>
#include <memory>
#include <cuda_runtime.h>
#include "../include/common.h"
#include "models/material_db.h"
#include "io/input_parser.h"
#include "visualization/vtk_writer.h"

class ThermoMechSimManager {
public:
    ThermoMechSimManager();
    ~ThermoMechSimManager();
    
    // Initialize the simulation from config file
    bool initialize(const std::string& config_file);
    
    // Run the simulation
    bool run();
    
    // Post-process results
    bool generateReport(const std::string& output_file);
    
    // Utility functions
    void printSimulationInfo() const;
    float getMaxStress() const;
    float getMaxTemperature() const;
    
private:
    // Configuration
    SimulationConfig config;
    PackageConfig package;
    
    // Material data
    std::unique_ptr<MaterialDatabase> material_db;
    std::vector<int> material_grid;
    
    // Simulation data
    std::vector<float> temperature;
    std::vector<float> stress;
    std::vector<float> power_map;
    
    // CUDA resources
    float* d_temperature;
    float* d_new_temperature;
    float* d_stress;
    float* d_power_map;
    int* d_material_grid;
    int* d_critical_points;
    
    // Helpers
    std::unique_ptr<InputParser> input_parser;
    std::unique_ptr<VtkWriter> vtk_writer;
    
    std::vector<cudaStream_t> streams;
    
    // Helper methods
    bool allocateDeviceMemory();
    bool freeDeviceMemory();
    bool copyDataToDevice();
    bool copyDataFromDevice();
    bool setupCudaDevice();
    bool timestep(int step);
    
    // Analysis functions
    bool identifyCriticalRegions();
    bool predictFailureRisks();
};

#endif // THERMO_MECH_SIM_MANAGER_H 