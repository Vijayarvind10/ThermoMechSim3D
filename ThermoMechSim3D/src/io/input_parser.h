#ifndef INPUT_PARSER_H
#define INPUT_PARSER_H

#include <string>
#include <vector>
#include <unordered_map>
#include "../include/common.h"

class InputParser {
public:
    InputParser();
    ~InputParser();
    
    // Parse configuration files
    bool parseSimulationConfig(const std::string& filename, SimulationConfig& config);
    bool parsePackageConfig(const std::string& filename, PackageConfig& package);
    
    // Load power maps and material data
    bool loadPowerMap(const std::string& filename, std::vector<float>& power_map,
                     int dim_x, int dim_y, int dim_z);
    bool loadMaterialDatabase(const std::string& filename, 
                             std::unordered_map<std::string, Material>& materials);
    
    // Convert stackup to simulation grid
    bool createMaterialGrid(const PackageConfig& package,
                          const std::unordered_map<std::string, Material>& materials,
                          std::vector<int>& material_grid,
                          int dim_x, int dim_y, int dim_z);
    
private:
    // Helper methods
    bool isJsonFile(const std::string& filename) const;
    bool isCsvFile(const std::string& filename) const;
    
    // Json parsing helpers
    bool parseJsonSimConfig(const std::string& filename, SimulationConfig& config);
    bool parseJsonPackage(const std::string& filename, PackageConfig& package);
    
    // CSV parsing helpers
    bool parseCsvPowerMap(const std::string& filename, std::vector<float>& power_map,
                         int dim_x, int dim_y, int dim_z);
    bool parseCsvMaterialDb(const std::string& filename,
                          std::unordered_map<std::string, Material>& materials);
};

#endif // INPUT_PARSER_H 