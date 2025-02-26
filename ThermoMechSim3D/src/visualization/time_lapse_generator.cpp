#include "time_lapse_generator.h"
#include <iostream>
#include <filesystem>
#include <cstdlib>
#include <sstream>

namespace fs = std::filesystem;

TimeLapseGenerator::TimeLapseGenerator() {
}

TimeLapseGenerator::~TimeLapseGenerator() {
}

bool TimeLapseGenerator::initialize(const std::string& output_dir) {
    this->output_dir = output_dir;
    
    // Create output directory if it doesn't exist
    try {
        fs::create_directories(output_dir);
        fs::create_directories(output_dir + "/frames");
    } catch (const std::exception& e) {
        std::cerr << "Error creating output directories: " << e.what() << std::endl;
        return false;
    }
    
    // Clear any existing frame files
    frame_files.clear();
    
    return true;
}

bool TimeLapseGenerator::addFrame(const std::string& vtk_file) {
    // Generate image file name
    std::string img_file = output_dir + "/frames/frame_" 
                         + std::to_string(frame_files.size()) + ".png";
    
    // Convert VTK to image
    if (!convertVtkToImage(vtk_file, img_file)) {
        return false;
    }
    
    // Add to frame list
    frame_files.push_back(img_file);
    return true;
}

bool TimeLapseGenerator::generateVideo(const std::string& output_file, int fps) {
    if (frame_files.empty()) {
        std::cerr << "No frames to generate video" << std::endl;
        return false;
    }
    
    std::cout << "Generating video from " << frame_files.size() << " frames..." << std::endl;
    
    // Build ffmpeg command
    std::stringstream command;
    command << "ffmpeg -y -framerate " << fps 
            << " -pattern_type glob -i \"" << output_dir << "/frames/frame_*.png\" "
            << "-c:v libx264 -pix_fmt yuv420p " << output_file;
    
    return executeCommand(command.str());
}

bool TimeLapseGenerator::generateGif(const std::string& output_file, int fps) {
    if (frame_files.empty()) {
        std::cerr << "No frames to generate GIF" << std::endl;
        return false;
    }
    
    std::cout << "Generating GIF from " << frame_files.size() << " frames..." << std::endl;
    
    // Build ImageMagick command
    std::stringstream command;
    command << "convert -delay " << (100 / fps) 
            << " -loop 0 \"" << output_dir << "/frames/frame_*.png\" "
            << output_file;
    
    return executeCommand(command.str());
}

bool TimeLapseGenerator::convertVtkToImage(const std::string& vtk_file, const std::string& img_file) {
    // In a real implementation, this would call Paraview's pvpython or similar tool
    // Here we'll simulate the conversion with a message
    
    std::cout << "Converting VTK file " << vtk_file << " to image " << img_file << std::endl;
    
    // For demo purposes, instead of actual conversion, create a dummy image:
    std::stringstream dummy_command;
    dummy_command << "echo 'Dummy image for " << vtk_file 
                 << "' > " << img_file;
    
    return executeCommand(dummy_command.str());
}

bool TimeLapseGenerator::executeCommand(const std::string& command) {
    std::cout << "Executing: " << command << std::endl;
    int result = std::system(command.c_str());
    
    if (result != 0) {
        std::cerr << "Command failed with exit code " << result << std::endl;
        return false;
    }
    
    return true;
} 