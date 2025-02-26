#include "vtk_writer.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

VtkWriter::VtkWriter() : dim_x(0), dim_y(0), dim_z(0), dx(0.0f), dy(0.0f), dz(0.0f) {
}

VtkWriter::~VtkWriter() {
}

bool VtkWriter::initialize(int dim_x, int dim_y, int dim_z, 
                         float dx, float dy, float dz, 
                         const std::string& output_dir) {
    this->dim_x = dim_x;
    this->dim_y = dim_y;
    this->dim_z = dim_z;
    this->dx = dx;
    this->dy = dy;
    this->dz = dz;
    this->output_dir = output_dir;
    
    // Create output directory if it doesn't exist
    try {
        fs::create_directories(output_dir);
    } catch (const std::exception& e) {
        std::cerr << "Error creating output directory: " << e.what() << std::endl;
        return false;
    }
    
    return true;
}

bool VtkWriter::writeTemperatureField(const std::vector<float>& temperature, 
                                    const std::string& filename, 
                                    int time_step) {
    std::string filepath = output_dir + "/" + filename + "_" + std::to_string(time_step) + ".vtk";
    std::ofstream file(filepath);
    
    if (!file) {
        std::cerr << "Failed to open file for writing: " << filepath << std::endl;
        return false;
    }
    
    // Write VTK header
    writeHeader(file, time_step);
    
    // Write coordinates
    writeCoordinates(file);
    
    // Write temperature data
    writeScalarData(file, temperature, "temperature");
    
    std::cout << "Temperature field written to " << filepath << std::endl;
    return true;
}

bool VtkWriter::writeStressField(const std::vector<float>& stress, 
                               const std::string& filename, 
                               int time_step) {
    std::string filepath = output_dir + "/" + filename + "_" + std::to_string(time_step) + ".vtk";
    std::ofstream file(filepath);
    
    if (!file) {
        std::cerr << "Failed to open file for writing: " << filepath << std::endl;
        return false;
    }
    
    // Write VTK header
    writeHeader(file, time_step);
    
    // Write coordinates
    writeCoordinates(file);
    
    // Write stress data
    writeScalarData(file, stress, "stress");
    
    std::cout << "Stress field written to " << filepath << std::endl;
    return true;
}

bool VtkWriter::writeFields(const std::vector<float>& temperature, 
                          const std::vector<float>& stress,
                          const std::vector<int>& material_ids,
                          const std::string& filename,
                          int time_step) {
    std::string filepath = output_dir + "/" + filename + "_" + std::to_string(time_step) + ".vtk";
    std::ofstream file(filepath);
    
    if (!file) {
        std::cerr << "Failed to open file for writing: " << filepath << std::endl;
        return false;
    }
    
    // Write VTK header
    writeHeader(file, time_step);
    
    // Write coordinates
    writeCoordinates(file);
    
    // Write temperature data
    writeScalarData(file, temperature, "temperature");
    
    // Write stress data
    writeScalarData(file, stress, "stress");
    
    // Convert material IDs to float for VTK
    std::vector<float> material_float(material_ids.size());
    for (size_t i = 0; i < material_ids.size(); ++i) {
        material_float[i] = static_cast<float>(material_ids[i]);
    }
    
    // Write material data
    writeScalarData(file, material_float, "material");
    
    std::cout << "Combined fields written to " << filepath << std::endl;
    return true;
}

bool VtkWriter::writeCriticalPoints(const std::vector<int>& critical_points,
                                  int num_points,
                                  const std::string& filename) {
    std::string filepath = output_dir + "/" + filename + ".vtk";
    std::ofstream file(filepath);
    
    if (!file) {
        std::cerr << "Failed to open file for writing: " << filepath << std::endl;
        return false;
    }
    
    // Write VTK header for points
    file << "# vtk DataFile Version 3.0\n";
    file << "Critical Stress Points\n";
    file << "ASCII\n";
    file << "DATASET POLYDATA\n";
    
    // Write point coordinates
    file << "POINTS " << num_points << " float\n";
    
    for (int i = 0; i < num_points; ++i) {
        int x = critical_points[i * 4];
        int y = critical_points[i * 4 + 1];
        int z = critical_points[i * 4 + 2];
        
        float x_pos = x * dx;
        float y_pos = y * dy;
        float z_pos = z * dz;
        
        file << x_pos << " " << y_pos << " " << z_pos << "\n";
    }
    
    // Write point data
    file << "POINT_DATA " << num_points << "\n";
    file << "SCALARS stress float 1\n";
    file << "LOOKUP_TABLE default\n";
    
    for (int i = 0; i < num_points; ++i) {
        float stress = *reinterpret_cast<float*>(&critical_points[i * 4 + 3]);
        file << stress << "\n";
    }
    
    std::cout << "Critical points written to " << filepath << std::endl;
    return true;
}

void VtkWriter::writeHeader(std::ofstream& file, int time_step) {
    file << "# vtk DataFile Version 3.0\n";
    file << "ThermoMechSim3D Output - Time Step " << time_step << "\n";
    file << "ASCII\n";
    file << "DATASET STRUCTURED_POINTS\n";
    file << "DIMENSIONS " << dim_x << " " << dim_y << " " << dim_z << "\n";
    file << "ORIGIN 0 0 0\n";
    file << "SPACING " << dx << " " << dy << " " << dz << "\n";
    file << "POINT_DATA " << dim_x * dim_y * dim_z << "\n";
}

void VtkWriter::writeCoordinates(std::ofstream& file) {
    // This function is not needed for STRUCTURED_POINTS format
    // as the coordinates are defined by origin and spacing
}

void VtkWriter::writeScalarData(std::ofstream& file, const std::vector<float>& data, 
                              const std::string& name) {
    file << "SCALARS " << name << " float 1\n";
    file << "LOOKUP_TABLE default\n";
    
    for (size_t i = 0; i < data.size(); ++i) {
        file << data[i] << "\n";
        
        // Add newline every 10 values for readability
        if ((i + 1) % 10 == 0) {
            file << "\n";
        }
    }
    file << "\n";
}

void VtkWriter::writeVectorData(std::ofstream& file, const std::vector<float>& data_x,
                              const std::vector<float>& data_y,
                              const std::vector<float>& data_z,
                              const std::string& name) {
    file << "VECTORS " << name << " float\n";
    
    for (size_t i = 0; i < data_x.size(); ++i) {
        file << data_x[i] << " " << data_y[i] << " " << data_z[i] << "\n";
        
        // Add newline every 5 values for readability
        if ((i + 1) % 5 == 0) {
            file << "\n";
        }
    }
    file << "\n";
} 