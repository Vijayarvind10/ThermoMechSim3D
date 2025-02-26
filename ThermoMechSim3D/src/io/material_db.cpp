#include "../models/material_db.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cuda_runtime.h>

MaterialDatabase::MaterialDatabase() {
    // Initialize with common semiconductor categories
    categories = {
        "semiconductor",
        "metal",
        "dielectric",
        "polymer",
        "thermal_interface",
        "substrate"
    };
}

MaterialDatabase::~MaterialDatabase() {
    // Clean up any resources
}

bool MaterialDatabase::initialize(const std::string& filename) {
    // Clear existing materials
    materials.clear();
    
    // Add default materials first
    addDefaultMaterials();
    
    // If filename is provided, load additional materials
    if (!filename.empty()) {
        if (!loadFromFile(filename)) {
            std::cerr << "Warning: Failed to load materials from " << filename << std::endl;
            std::cerr << "Using default materials only." << std::endl;
        }
    }
    
    std::cout << "Material database initialized with " << materials.size() << " materials." << std::endl;
    return true;
}

void MaterialDatabase::addDefaultMaterials() {
    // Silicon (Si)
    Material si;
    si.name = "Si";
    si.category = "semiconductor";
    si.thermal_conductivity = 149.0f;  // W/(m·K)
    si.specific_heat = 700.0f;         // J/(kg·K)
    si.density = 2329.0f;              // kg/m³
    si.youngs_modulus = 170.0e9f;      // Pa
    si.poissons_ratio = 0.28f;
    si.thermal_expansion = 2.6e-6f;    // 1/K
    si.yield_strength = 7000.0e6f;     // Pa
    si.ultimate_strength = 7000.0e6f;  // Pa
    si.melting_point = 1687.0f;        // K
    si.fracture_toughness = 0.9e6f;    // Pa·m^(1/2)
    si.electrical_resistivity = 640.0f; // Ω·m
    si.fatigue_coefficient = 1.0f;
    si.fatigue_exponent = 2.0f;
    si.source = "Default";
    materials["Si"] = si;
    
    // Copper (Cu)
    Material cu;
    cu.name = "Cu";
    cu.category = "metal";
    cu.thermal_conductivity = 400.0f;  // W/(m·K)
    cu.specific_heat = 385.0f;         // J/(kg·K)
    cu.density = 8960.0f;              // kg/m³
    cu.youngs_modulus = 130.0e9f;      // Pa
    cu.poissons_ratio = 0.34f;
    cu.thermal_expansion = 16.5e-6f;   // 1/K
    cu.yield_strength = 70.0e6f;       // Pa
    cu.ultimate_strength = 220.0e6f;   // Pa
    cu.melting_point = 1358.0f;        // K
    cu.fracture_toughness = 30.0e6f;   // Pa·m^(1/2)
    cu.electrical_resistivity = 1.68e-8f; // Ω·m
    cu.fatigue_coefficient = 0.9f;
    cu.fatigue_exponent = 1.5f;
    cu.source = "Default";
    materials["Cu"] = cu;
    
    // Copper TSV
    Material cu_tsv = cu;
    cu_tsv.name = "Cu_TSV";
    cu_tsv.category = "metal";
    cu_tsv.youngs_modulus = 110.0e9f;  // Lower due to deposition process
    cu_tsv.yield_strength = 50.0e6f;   // Lower due to confinement
    materials["Cu_TSV"] = cu_tsv;
    
    // Silicon Dioxide (SiO2)
    Material sio2;
    sio2.name = "SiO2";
    sio2.category = "dielectric";
    sio2.thermal_conductivity = 1.4f;  // W/(m·K)
    sio2.specific_heat = 730.0f;       // J/(kg·K)
    sio2.density = 2200.0f;            // kg/m³
    sio2.youngs_modulus = 70.0e9f;     // Pa
    sio2.poissons_ratio = 0.17f;
    sio2.thermal_expansion = 0.5e-6f;  // 1/K
    sio2.yield_strength = 8400.0e6f;   // Pa (brittle)
    sio2.ultimate_strength = 8400.0e6f; // Pa
    sio2.melting_point = 1986.0f;      // K
    sio2.fracture_toughness = 0.77e6f; // Pa·m^(1/2)
    sio2.electrical_resistivity = 1.0e16f; // Ω·m
    sio2.fatigue_coefficient = 1.0f;
    sio2.fatigue_exponent = 3.0f;
    sio2.source = "Default";
    materials["SiO2"] = sio2;
    
    // Aluminum (Al)
    Material al;
    al.name = "Al";
    al.category = "metal";
    al.thermal_conductivity = 237.0f;  // W/(m·K)
    al.specific_heat = 897.0f;         // J/(kg·K)
    al.density = 2700.0f;              // kg/m³
    al.youngs_modulus = 70.0e9f;       // Pa
    al.poissons_ratio = 0.35f;
    al.thermal_expansion = 23.1e-6f;   // 1/K
    al.yield_strength = 35.0e6f;       // Pa
    al.ultimate_strength = 90.0e6f;    // Pa
    al.melting_point = 933.0f;         // K
    al.fracture_toughness = 24.0e6f;   // Pa·m^(1/2)
    al.electrical_resistivity = 2.82e-8f; // Ω·m
    al.fatigue_coefficient = 0.85f;
    al.fatigue_exponent = 1.3f;
    al.source = "Default";
    materials["Al"] = al;
    
    // Underfill epoxy
    Material underfill;
    underfill.name = "Underfill";
    underfill.category = "polymer";
    underfill.thermal_conductivity = 0.3f;  // W/(m·K)
    underfill.specific_heat = 1100.0f;      // J/(kg·K)
    underfill.density = 1200.0f;            // kg/m³
    underfill.youngs_modulus = 8.5e9f;      // Pa
    underfill.poissons_ratio = 0.35f;
    underfill.thermal_expansion = 30.0e-6f;  // 1/K
    underfill.yield_strength = 50.0e6f;     // Pa
    underfill.ultimate_strength = 70.0e6f;  // Pa
    underfill.melting_point = 473.0f;       // K (glass transition)
    underfill.fracture_toughness = 0.5e6f;  // Pa·m^(1/2)
    underfill.electrical_resistivity = 1.0e14f; // Ω·m
    underfill.fatigue_coefficient = 0.7f;
    underfill.fatigue_exponent = 2.5f;
    underfill.source = "Default";
    materials["Underfill"] = underfill;
    
    // Thermal Interface Material (TIM)
    Material tim;
    tim.name = "TIM";
    tim.category = "thermal_interface";
    tim.thermal_conductivity = 5.0f;   // W/(m·K)
    tim.specific_heat = 1000.0f;       // J/(kg·K)
    tim.density = 2500.0f;             // kg/m³
    tim.youngs_modulus = 5.0e9f;       // Pa
    tim.poissons_ratio = 0.35f;
    tim.thermal_expansion = 25.0e-6f;  // 1/K
    tim.yield_strength = 5.0e6f;       // Pa
    tim.ultimate_strength = 7.0e6f;    // Pa
    tim.melting_point = 423.0f;        // K
    tim.fracture_toughness = 0.3e6f;   // Pa·m^(1/2)
    tim.electrical_resistivity = 1.0e10f; // Ω·m
    tim.fatigue_coefficient = 0.5f;
    tim.fatigue_exponent = 1.8f;
    tim.source = "Default";
    materials["TIM"] = tim;
    
    // Lead-free solder (SAC305)
    Material solder;
    solder.name = "SAC305";
    solder.category = "metal";
    solder.thermal_conductivity = 58.0f;  // W/(m·K)
    solder.specific_heat = 232.0f;        // J/(kg·K)
    solder.density = 7400.0f;             // kg/m³
    solder.youngs_modulus = 51.0e9f;      // Pa
    solder.poissons_ratio = 0.36f;
    solder.thermal_expansion = 21.0e-6f;  // 1/K
    solder.yield_strength = 32.0e6f;      // Pa
    solder.ultimate_strength = 48.0e6f;   // Pa
    solder.melting_point = 490.0f;        // K
    solder.fracture_toughness = 1.8e6f;   // Pa·m^(1/2)
    solder.electrical_resistivity = 1.3e-7f; // Ω·m
    solder.fatigue_coefficient = 0.45f;
    solder.fatigue_exponent = 1.2f;
    solder.source = "Default";
    materials["SAC305"] = solder;
}

bool MaterialDatabase::loadFromFile(const std::string& filename) {
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
        Material material;
        if (parseMaterialEntry(line, material)) {
            materials[material.name] = material;
            count++;
        }
    }
    
    std::cout << "Loaded " << count << " materials from " << filename << std::endl;
    return count > 0;
}

bool MaterialDatabase::parseMaterialEntry(const std::string& line, Material& material) {
    std::istringstream iss(line);
    std::string token;
    std::vector<std::string> tokens;
    
    while (std::getline(iss, token, ',')) {
        tokens.push_back(token);
    }
    
    if (tokens.size() < 15) {
        std::cerr << "Invalid material entry: " << line << std::endl;
        return false;
    }
    
    try {
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
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error parsing material entry: " << e.what() << std::endl;
        return false;
    }
}

bool MaterialDatabase::getMaterial(const std::string& name, Material& material) const {
    auto it = materials.find(name);
    if (it == materials.end()) {
        return false;
    }
    
    material = it->second;
    return true;
}

void MaterialDatabase::getMaterialsMap(std::unordered_map<std::string, Material>& out_materials) const {
    out_materials = materials;
}

std::vector<Material> MaterialDatabase::getMaterialsByCategory(const std::string& category) const {
    std::vector<Material> result;
    
    for (const auto& entry : materials) {
        if (entry.second.category == category) {
            result.push_back(entry.second);
        }
    }
    
    return result;
}

int MaterialDatabase::getMaterialCount() const {
    return materials.size();
}

std::vector<std::string> MaterialDatabase::getMaterialNames() const {
    std::vector<std::string> names;
    names.reserve(materials.size());
    
    for (const auto& entry : materials) {
        names.push_back(entry.first);
    }
    
    return names;
}

std::vector<std::string> MaterialDatabase::getCategories() const {
    return categories;
}

bool MaterialDatabase::createCudaMaterialTexture(float** device_material_data) {
    // Create a linear array of all material properties
    int num_materials = materials.size();
    int props_per_material = MATERIAL_PROPS_SIZE;
    
    std::vector<float> host_material_data(num_materials * props_per_material, 0.0f);
    
    // Map material names to indices
    std::unordered_map<std::string, int> material_indices;
    int idx = 0;
    
    for (const auto& entry : materials) {
        material_indices[entry.first] = idx;
        
        const Material& mat = entry.second;
        host_material_data[idx * props_per_material + PROP_CONDUCTIVITY] = mat.thermal_conductivity;
        host_material_data[idx * props_per_material + PROP_DENSITY] = mat.density;
        host_material_data[idx * props_per_material + PROP_SPECIFIC_HEAT] = mat.specific_heat;
        host_material_data[idx * props_per_material + PROP_YOUNGS_MODULUS] = mat.youngs_modulus;
        host_material_data[idx * props_per_material + PROP_POISSONS_RATIO] = mat.poissons_ratio;
        host_material_data[idx * props_per_material + PROP_THERMAL_EXPANSION] = mat.thermal_expansion;
        host_material_data[idx * props_per_material + PROP_YIELD_STRENGTH] = mat.yield_strength;
        host_material_data[idx * props_per_material + PROP_ULTIMATE_STRENGTH] = mat.ultimate_strength;
        host_material_data[idx * props_per_material + PROP_MELTING_POINT] = mat.melting_point;
        host_material_data[idx * props_per_material + PROP_FRACTURE_TOUGHNESS] = mat.fracture_toughness;
        
        idx++;
    }
    
    // Allocate device memory for material data
    cudaError_t error = cudaMalloc(device_material_data, 
                                  host_material_data.size() * sizeof(float));
    if (error != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for material data: " 
                  << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    // Copy material data to device
    error = cudaMemcpy(*device_material_data, host_material_data.data(),
                      host_material_data.size() * sizeof(float),
                      cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        std::cerr << "Failed to copy material data to device: " 
                  << cudaGetErrorString(error) << std::endl;
        cudaFree(*device_material_data);
        *device_material_data = nullptr;
        return false;
    }
    
    // Bind to texture
    cudaChannelFormatDesc channelDesc = 
        cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    
    cudaBindTexture(0, materialTex, *device_material_data, channelDesc,
                   host_material_data.size() * sizeof(float));
    
    return true;
} 