#include "ThermoMechSimManager.h"
#include "cuda/thermal_stress.h"
#include <fstream>
#include <iomanip>
#include <chrono>
#include <algorithm>

ThermoMechSimManager::ThermoMechSimManager()
    : d_temperature(nullptr),
      d_new_temperature(nullptr),
      d_stress(nullptr),
      d_power_map(nullptr),
      d_material_grid(nullptr),
      d_critical_points(nullptr) {
    
    material_db = std::make_unique<MaterialDatabase>();
    input_parser = std::make_unique<InputParser>();
    vtk_writer = std::make_unique<VtkWriter>();
}

ThermoMechSimManager::~ThermoMechSimManager() {
    // Clean up CUDA streams
    for (auto& stream : streams) {
        cudaStreamDestroy(stream);
    }
    
    // Free device memory
    freeDeviceMemory();
}

bool ThermoMechSimManager::initialize(const std::string& config_file) {
    // Parse simulation configuration
    if (!input_parser->parseSimulationConfig(config_file, config)) {
        std::cerr << "Failed to parse simulation config from: " << config_file << std::endl;
        return false;
    }
    
    // Parse package configuration
    if (!input_parser->parsePackageConfig(config.package_template, package)) {
        std::cerr << "Failed to parse package config from: " << config.package_template << std::endl;
        return false;
    }
    
    // Initialize material database
    if (!material_db->initialize(config.material_db_file)) {
        std::cerr << "Failed to initialize material database from: " << config.material_db_file << std::endl;
        return false;
    }
    
    // Initialize visualization
    if (!vtk_writer->initialize(config.dim_x, config.dim_y, config.dim_z, 
                               config.dx, config.dy, config.dz, 
                               config.output_dir)) {
        std::cerr << "Failed to initialize VTK writer" << std::endl;
        return false;
    }
    
    // Set up the material grid based on the stackup
    std::unordered_map<std::string, Material> materials;
    material_db->getMaterialsMap(materials);
    
    material_grid.resize(config.dim_x * config.dim_y * config.dim_z, 0);
    if (!input_parser->createMaterialGrid(package, materials, material_grid, 
                                         config.dim_x, config.dim_y, config.dim_z)) {
        std::cerr << "Failed to create material grid" << std::endl;
        return false;
    }
    
    // Load power map if specified
    if (!config.power_map_file.empty()) {
        power_map.resize(config.dim_x * config.dim_y * config.dim_z, 0.0f);
        if (!input_parser->loadPowerMap(config.power_map_file, power_map,
                                       config.dim_x, config.dim_y, config.dim_z)) {
            std::cerr << "Failed to load power map from: " << config.power_map_file << std::endl;
            return false;
        }
    } else {
        // Generate power map from individual die power maps in the stackup
        power_map.resize(config.dim_x * config.dim_y * config.dim_z, 0.0f);
        for (const auto& layer : package.stackup) {
            if (!layer.power_map_file.empty()) {
                std::vector<float> layer_power;
                layer_power.resize(config.dim_x * config.dim_y * config.dim_z, 0.0f);
                if (input_parser->loadPowerMap(layer.power_map_file, layer_power,
                                            config.dim_x, config.dim_y, config.dim_z)) {
                    // Combine the layer power with global power map
                    for (size_t i = 0; i < power_map.size(); ++i) {
                        power_map[i] += layer_power[i];
                    }
                } else {
                    std::cerr << "Warning: Failed to load power map for layer: " 
                              << layer.power_map_file << std::endl;
                }
            }
        }
    }
    
    // Initialize the temperature field with ambient temperature
    temperature.resize(config.dim_x * config.dim_y * config.dim_z, config.ambient_temp);
    stress.resize(config.dim_x * config.dim_y * config.dim_z, 0.0f);
    
    // Set up CUDA device and allocate memory
    if (!setupCudaDevice()) {
        std::cerr << "Failed to set up CUDA device" << std::endl;
        return false;
    }
    
    if (!allocateDeviceMemory()) {
        std::cerr << "Failed to allocate device memory" << std::endl;
        return false;
    }
    
    // Copy initial data to device
    if (!copyDataToDevice()) {
        std::cerr << "Failed to copy data to device" << std::endl;
        return false;
    }
    
    return true;
}

bool ThermoMechSimManager::setupCudaDevice() {
    cudaError_t error;
    
    // Set device
    error = cudaSetDevice(config.device_id);
    if (error != cudaSuccess) {
        std::cerr << "Failed to set CUDA device: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    // Create streams for async operations
    streams.resize(config.num_streams);
    for (int i = 0; i < config.num_streams; ++i) {
        error = cudaStreamCreate(&streams[i]);
        if (error != cudaSuccess) {
            std::cerr << "Failed to create CUDA stream: " << cudaGetErrorString(error) << std::endl;
            return false;
        }
    }
    
    return true;
}

bool ThermoMechSimManager::allocateDeviceMemory() {
    cudaError_t error;
    size_t size = config.dim_x * config.dim_y * config.dim_z * sizeof(float);
    
    // Allocate temperature fields
    error = cudaMalloc(&d_temperature, size);
    if (error != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for temperature: " 
                  << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    error = cudaMalloc(&d_new_temperature, size);
    if (error != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for new temperature: " 
                  << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    // Allocate stress field
    error = cudaMalloc(&d_stress, size);
    if (error != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for stress: " 
                  << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    // Allocate power map
    error = cudaMalloc(&d_power_map, size);
    if (error != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for power map: " 
                  << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    // Allocate material grid
    size_t material_size = config.dim_x * config.dim_y * config.dim_z * sizeof(MaterialPoint);
    error = cudaMalloc(&d_material_grid, material_size);
    if (error != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for material grid: " 
                  << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    // Allocate memory for critical points
    size_t critical_size = MAX_CRITICAL_POINTS * 4 * sizeof(int); // x, y, z, stress
    error = cudaMalloc(&d_critical_points, critical_size);
    if (error != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for critical points: " 
                  << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    return true;
}

bool ThermoMechSimManager::freeDeviceMemory() {
    if (d_temperature) cudaFree(d_temperature);
    if (d_new_temperature) cudaFree(d_new_temperature);
    if (d_stress) cudaFree(d_stress);
    if (d_power_map) cudaFree(d_power_map);
    if (d_material_grid) cudaFree(d_material_grid);
    if (d_critical_points) cudaFree(d_critical_points);
    
    d_temperature = nullptr;
    d_new_temperature = nullptr;
    d_stress = nullptr;
    d_power_map = nullptr;
    d_material_grid = nullptr;
    d_critical_points = nullptr;
    
    return true;
}

bool ThermoMechSimManager::copyDataToDevice() {
    cudaError_t error;
    
    // Copy temperature field
    error = cudaMemcpy(d_temperature, temperature.data(), 
                      temperature.size() * sizeof(float), 
                      cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        std::cerr << "Failed to copy temperature to device: " 
                  << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    // Copy power map
    if (!power_map.empty()) {
        error = cudaMemcpy(d_power_map, power_map.data(), 
                          power_map.size() * sizeof(float), 
                          cudaMemcpyHostToDevice);
        if (error != cudaSuccess) {
            std::cerr << "Failed to copy power map to device: " 
                      << cudaGetErrorString(error) << std::endl;
            return false;
        }
    }
    
    // Convert material grid to device format
    std::vector<MaterialPoint> material_points(config.dim_x * config.dim_y * config.dim_z);
    for (size_t i = 0; i < material_grid.size(); ++i) {
        material_points[i].material_id = material_grid[i];
        // Additional properties could be set here if needed
    }
    
    // Copy material grid
    error = cudaMemcpy(d_material_grid, material_points.data(), 
                      material_points.size() * sizeof(MaterialPoint), 
                      cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        std::cerr << "Failed to copy material grid to device: " 
                  << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    // Set up material texture memory
    float* d_material_props = nullptr;
    if (!material_db->createCudaMaterialTexture(&d_material_props)) {
        std::cerr << "Failed to create CUDA material texture" << std::endl;
        return false;
    }
    
    return true;
}

bool ThermoMechSimManager::copyDataFromDevice() {
    cudaError_t error;
    
    // Copy temperature field back to host
    error = cudaMemcpy(temperature.data(), d_temperature, 
                      temperature.size() * sizeof(float), 
                      cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        std::cerr << "Failed to copy temperature from device: " 
                  << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    // Copy stress field back to host
    error = cudaMemcpy(stress.data(), d_stress, 
                      stress.size() * sizeof(float), 
                      cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        std::cerr << "Failed to copy stress from device: " 
                  << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    return true;
}

bool ThermoMechSimManager::run() {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Main simulation loop
    for (int step = 0; step < config.num_steps; ++step) {
        if (!timestep(step)) {
            std::cerr << "Simulation failed at step " << step << std::endl;
            return false;
        }
        
        // Output results at specified intervals
        if (step % config.output_interval == 0) {
            // Copy data back to host for visualization
            if (!copyDataFromDevice()) {
                return false;
            }
            
            // Write data to VTK files
            std::string filename = "thermostress_" + std::to_string(step);
            vtk_writer->writeFields(temperature, stress, material_grid, filename, step);
            
            // Print progress
            auto current_time = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                current_time - start_time).count();
            
            float max_temp = *std::max_element(temperature.begin(), temperature.end());
            float max_stress_val = *std::max_element(stress.begin(), stress.end());
            
            std::cout << "Step " << step << "/" << config.num_steps 
                      << " (Time: " << elapsed << "s)" << std::endl;
            std::cout << "  Max Temperature: " << max_temp << " K" << std::endl;
            std::cout << "  Max von Mises Stress: " << max_stress_val / 1e6 << " MPa" << std::endl;
        }
        
        // Check for convergence
        if (step > 0 && step % 100 == 0) {
            float convergence = checkConvergence();
            if (convergence < config.convergence_threshold) {
                std::cout << "Simulation converged at step " << step 
                          << " (convergence = " << convergence << ")" << std::endl;
                break;
            }
        }
    }
    
    // Final analysis
    if (!identifyCriticalRegions()) {
        std::cerr << "Failed to identify critical regions" << std::endl;
    }
    
    // Generate failure predictions
    if (!predictFailureRisks()) {
        std::cerr << "Failed to predict failure risks" << std::endl;
    }
    
    // Output final results
    if (!copyDataFromDevice()) {
        return false;
    }
    
    // Final visualization
    vtk_writer->writeFields(temperature, stress, material_grid, "thermostress_final", config.num_steps);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        end_time - start_time).count();
    
    std::cout << "Simulation completed in " << elapsed << " seconds" << std::endl;
    
    return true;
}

bool ThermoMechSimManager::timestep(int step) {
    // Select a stream based on step
    cudaStream_t stream = streams[step % streams.size()];
    
    // Launch the kernel to compute the new temperature and stress
    launchThermalStressKernel(
        d_temperature,
        d_new_temperature,
        d_stress,
        reinterpret_cast<const MaterialPoint*>(d_material_grid),
        config.dt, config.dx, config.dy, config.dz,
        config.dim_x, config.dim_y, config.dim_z,
        config.ref_temp,
        d_power_map,
        stream
    );
    
    // Swap temperature buffers
    std::swap(d_temperature, d_new_temperature);
    
    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in timestep " << step << ": " 
                  << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    return true;
}

float ThermoMechSimManager::checkConvergence() {
    // Copy latest temperature to host
    std::vector<float> current_temp(config.dim_x * config.dim_y * config.dim_z);
    cudaMemcpy(current_temp.data(), d_temperature, 
              current_temp.size() * sizeof(float),
              cudaMemcpyDeviceToHost);
    
    // Calculate change from previous temperature
    float max_diff = 0.0f;
    for (size_t i = 0; i < temperature.size(); ++i) {
        float diff = std::abs(current_temp[i] - temperature[i]);
        max_diff = std::max(max_diff, diff);
    }
    
    // Update host temperature
    temperature = current_temp;
    
    return max_diff;
}

bool ThermoMechSimManager::identifyCriticalRegions() {
    // Find regions with stress above threshold
    int critical_count = findCriticalStressRegions(
        d_stress,
        d_critical_points,
        config.stress_threshold,
        config.dim_x, config.dim_y, config.dim_z,
        streams[0]
    );
    
    if (critical_count > 0) {
        std::cout << "Found " << critical_count << " critical stress regions" << std::endl;
        
        // Copy critical points back to host
        std::vector<int> critical_points(critical_count * 4);
        cudaMemcpy(critical_points.data(), d_critical_points,
                  critical_points.size() * sizeof(int),
                  cudaMemcpyDeviceToHost);
        
        // Write critical points to file
        vtk_writer->writeCriticalPoints(critical_points, critical_count, "critical_stress_regions");
        
        // Send warning to ATE if integrated
        sendWarningToATE(critical_count);
    } else {
        std::cout << "No critical stress regions found" << std::endl;
    }
    
    return true;
}

bool ThermoMechSimManager::predictFailureRisks() {
    // Calculate MTF (Mean Time to Failure) based on stress levels
    float max_stress_val = getMaxStress();
    float max_temp = getMaxTemperature();
    
    // Generate report
    std::ofstream report(config.output_dir + "/failure_risk_report.txt");
    if (!report) {
        std::cerr << "Failed to create failure risk report" << std::endl;
        return false;
    }
    
    report << "Failure Risk Assessment:\n";
    report << "------------------------\n\n";
    
    // Analyze TSV stress
    float tsv_margin = calculateTSVStressMargin();
    report << "- TSV Array: " << tsv_margin * 100.0f << "% stress margin\n";
    
    // Analyze microbump stress
    float microbump_margin = calculateMicrobumpStressMargin();
    report << "- Microbump Cluster: " << microbump_margin * 100.0f << "% margin";
    if (microbump_margin < 0.5f) {
        report << " (WARNING)\n";
    } else {
        report << "\n";
    }
    
    // Estimated lifetime
    float mtf_years = estimateMTF(max_temp, max_stress_val);
    report << "- Estimated MTF: " << mtf_years << " years @ " << max_temp << "°C\n\n";
    
    // Package-specific recommendations
    report << "Recommendations:\n";
    report << "----------------\n";
    if (microbump_margin < 0.5f) {
        report << "- Reduce temperature gradient across microbump interfaces\n";
        report << "- Consider thicker underfill layer for stress relief\n";
    }
    if (tsv_margin < 0.7f) {
        report << "- Optimize TSV placement or reduce TSV density\n";
        report << "- Evaluate alternative TSV liner materials\n";
    }
    
    std::cout << "Failure risk assessment completed. Report saved to: " 
              << config.output_dir << "/failure_risk_report.txt" << std::endl;
    
    return true;
}

float ThermoMechSimManager::calculateTSVStressMargin() {
    // In a real implementation, this would analyze stress in TSV regions
    // For this example, we'll use a simplified calculation
    float tsv_stress = 0.0f;
    int tsv_count = 0;
    
    // Find regions with TSV material
    for (size_t i = 0; i < material_grid.size(); ++i) {
        // Assuming material ID for Cu_TSV is known
        if (material_grid[i] == 2) { // Assuming 2 is Cu_TSV
            tsv_stress += stress[i];
            tsv_count++;
        }
    }
    
    if (tsv_count == 0) return 1.0f; // No TSVs found
    
    float avg_tsv_stress = tsv_stress / tsv_count;
    
    // Get material yield strength for Cu_TSV
    Material tsv_material;
    material_db->getMaterial("Cu_TSV", tsv_material);
    
    // Calculate margin as percentage of yield strength
    return 1.0f - (avg_tsv_stress / tsv_material.yield_strength);
}

float ThermoMechSimManager::calculateMicrobumpStressMargin() {
    // In a real implementation, this would analyze stress in microbump regions
    // For this example, we'll use a simplified calculation
    
    // Microbumps are typically at interfaces between layers
    // Let's assume they're at the interfaces of every other layer
    float microbump_stress = 0.0f;
    int microbump_count = 0;
    
    int layer_height = config.dim_z / package.stackup.size();
    
    for (size_t layer = 1; layer < package.stackup.size(); layer += 2) {
        int z_start = layer * layer_height - 1;
        int z_end = z_start + 2; // Interface region
        
        for (int z = z_start; z <= z_end; z++) {
            for (int y = 0; y < config.dim_y; y++) {
                for (int x = 0; x < config.dim_x; x++) {
                    int idx = z * config.dim_x * config.dim_y + y * config.dim_x + x;
                    microbump_stress += stress[idx];
                    microbump_count++;
                }
            }
        }
    }
    
    if (microbump_count == 0) return 1.0f; // No microbumps found
    
    float avg_microbump_stress = microbump_stress / microbump_count;
    
    // Typical yield strength for SnAg microbumps
    float microbump_yield_strength = 30e6; // 30 MPa, simplified
    
    // Calculate margin as percentage of yield strength
    return 1.0f - (avg_microbump_stress / microbump_yield_strength);
}

float ThermoMechSimManager::estimateMTF(float max_temp, float max_stress) {
    // Simplified Black's equation for electromigration
    // MTF = A * (1/J)^n * exp(Ea/kT)
    
    // Constants
    float A = 1e5; // Scale factor
    float n = 2.0; // Current density exponent
    float Ea = 0.7; // Activation energy (eV)
    float k = 8.617e-5; // Boltzmann constant (eV/K)
    
    // Stress-based current density approximation
    float effective_current_density = max_stress / 1e7;
    
    // Temperature in Kelvin
    float T = max_temp;
    
    // Black's equation with stress modification
    float mtf_hours = A * pow(1.0/effective_current_density, n) * exp(Ea/(k*T));
    
    // Convert to years
    return mtf_hours / (24.0f * 365.0f);
}

void ThermoMechSimManager::sendWarningToATE(int critical_count) {
    // Interface with Automated Test Equipment
    // This is a placeholder for actual ATE integration
    std::cout << "ATE WARNING: " << critical_count << " stress points exceed threshold!" << std::endl;
    
    // In a real implementation, this would involve communicating with test equipment
    // Example:
    // if (critical_count > 10) {
    //     ate_send_command("BIN 5; LOG STRESS_OVER_95%");
    // }
}

bool ThermoMechSimManager::generateReport(const std::string& output_file) {
    std::ofstream report(output_file);
    if (!report) {
        std::cerr << "Failed to create simulation report" << std::endl;
        return false;
    }
    
    // Report header
    report << "ThermoMechSim3D Simulation Report\n";
    report << "================================\n\n";
    
    // Simulation parameters
    report << "Simulation Parameters:\n";
    report << "---------------------\n";
    report << "Grid dimensions: " << config.dim_x << " x " << config.dim_y << " x " << config.dim_z << "\n";
    report << "Grid spacing: " << config.dx * 1e6 << " x " << config.dy * 1e6 << " x " << config.dz * 1e6 << " µm\n";
    report << "Time step: " << config.dt << " s\n";
    report << "Number of steps: " << config.num_steps << "\n";
    report << "Reference temperature: " << config.ref_temp << " K\n";
    report << "Ambient temperature: " << config.ambient_temp << " K\n\n";
    
    // Package information
    report << "Package Information:\n";
    report << "-------------------\n";
    report << "Name: " << package.name << "\n";
    report << "Number of layers: " << package.stackup.size() << "\n";
    report << "Total thickness: " << getTotalStackThickness() * 1e6 << " µm\n\n";
    
    // Results summary
    report << "Results Summary:\n";
    report << "---------------\n";
    report << "Maximum temperature: " << getMaxTemperature() << " K\n";
    report << "Minimum temperature: " << getMinTemperature() << " K\n";
    report << "Temperature gradient: " << getMaxTemperature() - getMinTemperature() << " K\n";
    report << "Maximum von Mises stress: " << getMaxStress() / 1e6 << " MPa\n";
    report << "Critical stress regions: " << getNumCriticalRegions() << "\n\n";
    
    // Failure risk assessment
    report << "Failure Risk Assessment:\n";
    report << "------------------------\n";
    
    float tsv_margin = calculateTSVStressMargin();
    float microbump_margin = calculateMicrobumpStressMargin();
    float mtf_years = estimateMTF(getMaxTemperature(), getMaxStress());
    
    report << "TSV stress margin: " << tsv_margin * 100.0f << "%\n";
    report << "Microbump stress margin: " << microbump_margin * 100.0f << "%\n";
    report << "Estimated MTF: " << mtf_years << " years @ " << getMaxTemperature() << " K\n\n";
    
    // Output files
    report << "Output Files:\n";
    report << "------------\n";
    report << "Temperature and stress fields: " << config.output_dir << "/thermostress_*.vtk\n";
    report << "Critical regions: " << config.output_dir << "/critical_stress_regions.vtk\n";
    report << "Failure risk report: " << config.output_dir << "/failure_risk_report.txt\n";
    
    std::cout << "Simulation report generated: " << output_file << std::endl;
    
    return true;
}

void ThermoMechSimManager::printSimulationInfo() const {
    std::cout << "ThermoMechSim3D Simulation\n";
    std::cout << "=========================\n";
    std::cout << "Package: " << package.name << "\n";
    std::cout << "Grid: " << config.dim_x << " x " << config.dim_y << " x " << config.dim_z << "\n";
    std::cout << "Steps: " << config.num_steps << "\n";
    std::cout << "Output directory: " << config.output_dir << "\n";
    
    int num_materials = material_db->getMaterialCount();
    std::cout << "Material database: " << num_materials << " materials loaded\n";
    
    std::cout << "=========================\n";
}

float ThermoMechSimManager::getMaxStress() const {
    if (stress.empty()) return 0.0f;
    return *std::max_element(stress.begin(), stress.end());
}

float ThermoMechSimManager::getMaxTemperature() const {
    if (temperature.empty()) return 0.0f;
    return *std::max_element(temperature.begin(), temperature.end());
}

float ThermoMechSimManager::getMinTemperature() const {
    if (temperature.empty()) return 0.0f;
    return *std::min_element(temperature.begin(), temperature.end());
}

int ThermoMechSimManager::getNumCriticalRegions() const {
    int count = 0;
    for (float s : stress) {
        if (s > config.stress_threshold) {
            count++;
        }
    }
    return count;
}

float ThermoMechSimManager::getTotalStackThickness() const {
    float total = 0.0f;
    for (const auto& layer : package.stackup) {
        total += layer.thickness;
    }
    return total;
} 