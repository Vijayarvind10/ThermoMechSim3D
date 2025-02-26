#ifndef VTK_WRITER_H
#define VTK_WRITER_H

#include <string>
#include <vector>
#include "../include/common.h"

class VtkWriter {
public:
    VtkWriter();
    ~VtkWriter();
    
    // Initialize with simulation dimensions
    bool initialize(int dim_x, int dim_y, int dim_z, 
                   float dx, float dy, float dz,
                   const std::string& output_dir);
    
    // Write temperature field to VTK file
    bool writeTemperatureField(const std::vector<float>& temperature,
                              const std::string& filename,
                              int time_step);
    
    // Write stress field to VTK file
    bool writeStressField(const std::vector<float>& stress,
                         const std::string& filename,
                         int time_step);
    
    // Write multiple fields (temperature + stress)
    bool writeFields(const std::vector<float>& temperature,
                    const std::vector<float>& stress,
                    const std::vector<int>& material_ids,
                    const std::string& filename,
                    int time_step);
    
    // Write critical points for failure analysis
    bool writeCriticalPoints(const std::vector<int>& critical_points,
                            int num_points,
                            const std::string& filename);
    
private:
    int dim_x, dim_y, dim_z;
    float dx, dy, dz;
    std::string output_dir;
    
    // Helper methods
    void writeHeader(std::ofstream& file, int time_step);
    void writeCoordinates(std::ofstream& file);
    void writeScalarData(std::ofstream& file, const std::vector<float>& data, 
                        const std::string& name);
    void writeVectorData(std::ofstream& file, const std::vector<float>& data_x,
                        const std::vector<float>& data_y,
                        const std::vector<float>& data_z,
                        const std::string& name);
};

#endif // VTK_WRITER_H 