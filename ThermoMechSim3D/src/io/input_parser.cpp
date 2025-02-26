#include "input_parser.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <filesystem>

// Use C++17 filesystem or fallback to experimental if needed
namespace fs = std::filesystem;

InputParser::InputParser() {
    // Constructor
}

InputParser::~InputParser() {
    // Destructor
}

bool InputParser::parseSimulationConfig(const std::string& filename, SimulationConfig& config) {
    if (isJsonFile(filename)) {
        return parseJsonSimConfig(filename, config);
    } else {
        std::cerr << "Unsupported configuration file format: " << filename << std::endl;
        return false;
    }
}

bool InputParser::parsePackageConfig(const std::string& filename, PackageConfig& package) {
    if (isJsonFile(filename)) {
        return parseJsonPackage(filename, package);
    } else {
        std::cerr << "Unsupported package file format: " << filename << std::endl;
        return false;
    }
}

bool InputParser::loadPowerMap(const std::string& filename, std::vector<float>& power_map,
                             int dim_x, int dim_y, int dim_z) {
    if (isCsvFile(filename)) {
        return parseCsvPowerMap(filename, power_map, dim_x, dim_y, dim_z);
    } else {
        std::cerr << "Unsupported power map file format: " << filename << std::endl;
        return false;
    }
}

bool InputParser::loadMaterialDatabase(const std::string& filename,
                                     std::unordered_map<std::string, Material>& materials) {
    if (isCsvFile(filename)) {
        return parseCsvMaterialDb(filename, materials);
    } else {
        std::cerr << "Unsupported material database file format: " << filename << std::endl;
        return false;
    }
}

bool InputParser::isJsonFile(const std::string& filename) const {
    size_t pos = filename.find_last_of('.');
    if (pos == std::string::npos) return false;
    
    std::string extension = filename.substr(pos + 1);
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
    
    return extension == "json";
}

bool InputParser::isCsvFile(const std::string& filename) const {
    size_t pos = filename.find_last_of('.');
    if (pos == std::string::npos) return false;
    
    std::string extension = filename.substr(pos + 1);
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
    
    return extension == "csv";
}

bool InputParser::parseJsonSimConfig(const std::string& filename, SimulationConfig& config) {
    // In a real implementation, this would use a JSON library like nlohmann/json
    // For this example, we'll use a simplified approach
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Failed to open simulation config file: " << filename << std::endl;
        return false;
    }
    
    // For demonstration - in reality, use a JSON parser
    std::cout << "Parsing simulation config from " << filename << std::endl;
    
    // Set default values
    config.dim_x = 128;
    config.dim_y = 128;
    config.dim_z = 64;
    config.dx = 1e-6;  // 1 µm
    config.dy = 1e-6;
    config.dz = 1e-6;
    config.dt = 1e-9;  // 1 ns
    config.num_steps = 1000;
    config.output_interval = 100;
    config.ref_temp = 298.15;  // 25°C
    config.ambient_temp = 298.15;
    config.stress_threshold = 100e6;  // 100 MPa
    config.output_dir = "output";
    config.power_map_file = "";
    config.material_db_file = "materials.csv";
    config.package_template = "examples/hbm_stack.json";
    config.convergence_threshold = 1e-4;
    config.max_iterations = 10000;
    config.device_id = 0;
    config.use_async_transfers = true;
    config.num_streams = 4;
    
    // Create output directory if it doesn't exist
    fs::create_directories(config.output_dir);
    
    return true;
}

bool InputParser::parseJsonPackage(const std::string& filename, PackageConfig& package) {
    // In a real implementation, this would use a JSON library like nlohmann/json
    // For this example, we'll use a simplified approach with hardcoded values for HBM stack
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Failed to open package config file: " << filename << std::endl;
        return false;
    }
    
    // For demonstration - in reality, use a JSON parser
    std::cout << "Parsing package config from " << filename << std::endl;
    
    // HBM stack configuration (hardcoded for example)
    package.name = "High Bandwidth Memory Stack";
    package.ambient_temp = 298.15;
    package.convection_coefficient = 1e4;
    package.boundary_top = "convection";
    package.boundary_bottom = "constant_temp";
    package.boundary_sides = "adiabatic";
    
    // Create the stackup layers
    // Logic die
    Layer logic;
    logic.material_name = "Si";
    logic.thickness = 100e-6;
    logic.power_map_file = "hbm_logic_die.csv";
    logic.thermal_interface = false;
    package.stackup.push_back(logic);
    
    // TSV layer
    Layer tsv1;
    tsv1.material_name = "Cu_TSV";
    tsv1.thickness = 50e-6;
    tsv1.power_map_file = "";
    tsv1.thermal_interface = true;
    package.stackup.push_back(tsv1);
    
    // DRAM die 1
    Layer dram1;
    dram1.material_name = "Si";
    dram1.thickness = 50e-6;
    dram1.power_map_file = "hbm_dram_die1.csv";
    dram1.thermal_interface = false;
    package.stackup.push_back(dram1);
    
    // TSV layer
    Layer tsv2;
    tsv2.material_name = "Cu_TSV";
    tsv2.thickness = 50e-6;
    tsv2.power_map_file = "";
    tsv2.thermal_interface = true;
    package.stackup.push_back(tsv2);
    
    // DRAM die 2
    Layer dram2;
    dram2.material_name = "Si";
    dram2.thickness = 50e-6;
    dram2.power_map_file = "hbm_dram_die2.csv";
    dram2.thermal_interface = false;
    package.stackup.push_back(dram2);
    
    // TSV layer
    Layer tsv3;
    tsv3.material_name = "Cu_TSV";
    tsv3.thickness = 50e-6;
    tsv3.power_map_file = "";
    tsv3.thermal_interface = true;
    package.stackup.push_back(tsv3);
    
    // DRAM die 3
    Layer dram3;
    dram3.material_name = "Si";
    dram3.thickness = 50e-6;
    dram3.power_map_file = "hbm_dram_die3.csv";
    dram3.thermal_interface = false;
    package.stackup.push_back(dram3);
    
    // TSV layer
    Layer tsv4;
    tsv4.material_name = "Cu_TSV";
    tsv4.thickness = 50e-6;
    tsv4.power_map_file = "";
    tsv4.thermal_interface = true;
    package.stackup.push_back(tsv4);
    
    // DRAM die 4
    Layer dram4;
    dram4.material_name = "Si";
    dram4.thickness = 50e-6;
    dram4.power_map_file = "hbm_dram_die4.csv";
    dram4.thermal_interface = false;
    package.stackup.push_back(dram4);
    
    // Interposer
    Layer interposer;
    interposer.material_name = "SiO2";
    interposer.thickness = 20e-6;
    interposer.power_map_file = "";
    interposer.thermal_interface = true;
    package.stackup.push_back(interposer);
    
    return true;
}

bool InputParser::parseCsvPowerMap(const std::string& filename, std::vector<float>& power_map,
                                 int dim_x, int dim_y, int dim_z) {
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Failed to open power map file: " << filename << std::endl;
        return false;
    }
    
    // For this example, we'll generate a synthetic power map
    // In a real implementation, this would parse a CSV with actual power values
    std::cout << "Loading power map from " << filename << std::endl;
    
    // Generate synthetic power map
    // For example: concentrated heat in the center of the XY plane
    float total_power = 100.0f;  // Watts
    float center_x = dim_x / 2.0f;
    float center_y = dim_y / 2.0f;
    float sigma_sq = dim_x * dim_y / 100.0f;
    
    // Only apply power to the bottom 20% of the stack (logic die)
    int power_z_start = 0;
    int power_z_end = dim_z / 5;
    
    // Clear and resize power map
    power_map.clear();
    power_map.resize(dim_x * dim_y * dim_z, 0.0f);
    
    // Gaussian distribution centered on chip
    float total = 0.0f;
    for (int z = power_z_start; z < power_z_end; z++) {
        for (int y = 0; y < dim_y; y++) {
            for (int x = 0; x < dim_x; x++) {
                float dx = x - center_x;
                float dy = y - center_y;
                float r_sq = dx*dx + dy*dy;
                float power_density = exp(-r_sq / (2.0f * sigma_sq));
                
                int idx = z * dim_x * dim_y + y * dim_x + x;
                power_map[idx] = power_density;
                total += power_density;
            }
        }
    }
    
    // Normalize to get the desired total power
    if (total > 0.0f) {
        float scale = total_power / total;
        for (int z = power_z_start; z < power_z_end; z++) {
            for (int y = 0; y < dim_y; y++) {
                for (int x = 0; x < dim_x; x++) {
                    int idx = z * dim_x * dim_y + y * dim_x + x;
                    power_map[idx] *= scale;
                }
            }
        }
    }
    
    std::cout << "Generated synthetic power map with total power of " 
              << total_power << " W" << std::endl;
    
    return true;
}

bool InputParser::parseCsvMaterialDb(const std::string& filename,
                                   std::unordered_map<std::string, Material>& materials) {
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Failed to open material database file: " << filename << std::endl;
        return false;
    }
    
    std::string line;
    int count = 0;
    
    // Skip header line
    std::getline(file, line);
    
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string token;
        std::vector<std::string> tokens;
        
        while (std::getline(iss, token, ',')) {
            tokens.push_back(token);
        }
        
        if (tokens.size() < 15) {
            std::cerr << "Invalid material entry: " << line << std::endl;
            continue;
        }
        
        try {
            Material material;
            int idx = 0;
            material.name = tokens[idx++];
            material.category = tokens[idx++];
            material.thermal_conductivity = std::stof(tokens[idx++]);
            material.specific_heat = std::stof(tokens[idx++]);
            material.density = std::stof(tokens[idx++]);
            material.youngs_modulus = std::stof(tokens[idx++]);
            material.poissons_ratio = std::stof(tokens[idx++]);
            material.thermal_expansion = std::stof(tokens[idx++]);
            material.yield_strength = std::stof(tokens[idx++]);
            material.ultimate_strength = std::stof(tokens[idx++]);
            material.melting_point = std::stof(tokens[idx++]);
            material.fracture_toughness = std::stof(tokens[idx++]);
            material.electrical_resistivity = std::stof(tokens[idx++]);
            material.fatigue_coefficient = std::stof(tokens[idx++]);
            material.fatigue_exponent = std::stof(tokens[idx++]);
            
            if (idx < tokens.size()) {
                material.source = tokens[idx];
            } else {
                material.source = "CSV Database";
            }
            
            materials[material.name] = material;
            count++;
        } catch (const std::exception& e) {
            std::cerr << "Error parsing material entry: " << e.what() << std::endl;
        }
    }
    
    std::cout << "Loaded " << count << " materials from " << filename << std::endl;
    return count > 0;
}

bool InputParser::createMaterialGrid(const PackageConfig& package,
                                   const std::unordered_map<std::string, Material>& materials,
                                   std::vector<int>& material_grid,
                                   int dim_x, int dim_y, int dim_z) {
    // Map material names to indices
    std::unordered_map<std::string, int> material_indices;
    int idx = 0;
    
    for (const auto& entry : materials) {
        material_indices[entry.first] = idx++;
    }
    
    // Clear and resize material grid
    material_grid.clear();
    material_grid.resize(dim_x * dim_y * dim_z, 0);
    
    // Calculate total thickness
    float total_thickness = 0.0f;
    for (const auto& layer : package.stackup) {
        total_thickness += layer.thickness;
    }
    
    // Assign materials to grid cells based on z-position
    float z_pos = 0.0f;
    for (const auto& layer : package.stackup) {
        // Map layer to z-indices
        float layer_start = z_pos / total_thickness;
        z_pos += layer.thickness;
        float layer_end = z_pos / total_thickness;
        
        int z_start = static_cast<int>(layer_start * dim_z);
        int z_end = static_cast<int>(layer_end * dim_z);
        
        // Handle rounding issues for the last layer
        if (&layer == &package.stackup.back()) {
            z_end = dim_z;
        }
        
        // Assign material index to this layer
        if (material_indices.find(layer.material_name) != material_indices.end()) {
            int mat_idx = material_indices[layer.material_name];
            
            for (int z = z_start; z < z_end; z++) {
                for (int y = 0; y < dim_y; y++) {
                    for (int x = 0; x < dim_x; x++) {
                        int grid_idx = z * dim_x * dim_y + y * dim_x + x;
                        material_grid[grid_idx] = mat_idx;
                    }
                }
            }
        } else {
            std::cerr << "Warning: Material '" << layer.material_name 
                      << "' not found in material database" << std::endl;
        }
    }
    
    return true;
} 