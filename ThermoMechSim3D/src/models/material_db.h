#ifndef MATERIAL_DB_H
#define MATERIAL_DB_H

#include <string>
#include <unordered_map>
#include <vector>
#include "../include/common.h"

class MaterialDatabase {
public:
    MaterialDatabase();
    ~MaterialDatabase();
    
    // Initialize the database
    bool initialize(const std::string& filename);
    
    // Get material by name
    bool getMaterial(const std::string& name, Material& material) const;
    
    // Get all materials by category
    std::vector<Material> getMaterialsByCategory(const std::string& category) const;
    
    // Get total count of materials
    int getMaterialCount() const;
    
    // Create texture memory for CUDA
    bool createCudaMaterialTexture(float** device_material_data);
    
    // Utility functions
    std::vector<std::string> getMaterialNames() const;
    std::vector<std::string> getCategories() const;
    
private:
    std::unordered_map<std::string, Material> materials;
    std::vector<std::string> categories;
    
    // Add default materials
    void addDefaultMaterials();
    
    // Load materials from file
    bool loadFromFile(const std::string& filename);
    
    // Parse a single material entry
    bool parseMaterialEntry(const std::string& line, Material& material);
};

#endif // MATERIAL_DB_H 