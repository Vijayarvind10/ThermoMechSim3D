#ifndef TIME_LAPSE_GENERATOR_H
#define TIME_LAPSE_GENERATOR_H

#include <string>
#include <vector>

class TimeLapseGenerator {
public:
    TimeLapseGenerator();
    ~TimeLapseGenerator();
    
    // Initialize with output directory
    bool initialize(const std::string& output_dir);
    
    // Add a frame from VTK file
    bool addFrame(const std::string& vtk_file);
    
    // Generate time-lapse video
    bool generateVideo(const std::string& output_file, int fps = 10);
    
    // Generate animated GIF
    bool generateGif(const std::string& output_file, int fps = 10);
    
private:
    std::string output_dir;
    std::vector<std::string> frame_files;
    
    // Convert VTK to image (using external tools)
    bool convertVtkToImage(const std::string& vtk_file, const std::string& img_file);
    
    // Execute external command
    bool executeCommand(const std::string& command);
};

#endif // TIME_LAPSE_GENERATOR_H 