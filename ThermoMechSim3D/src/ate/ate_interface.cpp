#include "ate_interface.h"
#include <iostream>
#include <sstream>

ATEInterface::ATEInterface() : connected(false), port(0), socket_fd(-1) {
}

ATEInterface::~ATEInterface() {
    if (connected) {
        disconnect();
    }
}

bool ATEInterface::initialize(const std::string& ate_address, int port) {
    this->ate_address = ate_address;
    this->port = port;
    
    // In a real implementation, this would establish a socket connection
    // Here, we'll just mock the connection
    std::cout << "Connecting to ATE at " << ate_address << ":" << port << "..." << std::endl;
    
    // Simulate connection success
    connected = true;
    socket_fd = 100; // Mock file descriptor
    
    std::cout << "Connected to ATE system" << std::endl;
    return true;
}

bool ATEInterface::sendCommand(const std::string& command) {
    if (!checkConnection()) return false;
    
    std::cout << "Sending command to ATE: " << command << std::endl;
    
    // In a real implementation, this would send the command over the socket
    // For now, just simulate success
    return true;
}

bool ATEInterface::sendBinResult(int bin_number, const std::string& failure_reason) {
    if (!checkConnection()) return false;
    
    std::stringstream command;
    command << "BIN " << bin_number << "; LOG " << failure_reason;
    
    return sendCommand(command.str());
}

bool ATEInterface::sendStressData(const std::vector<float>& stress_map,
                                int dim_x, int dim_y, int dim_z) {
    if (!checkConnection()) return false;
    
    // In a real implementation, would pack and send the stress data
    // For now, just log that we're sending it
    std::cout << "Sending stress map data for " 
              << dim_x << "x" << dim_y << "x" << dim_z 
              << " grid to ATE" << std::endl;
    
    return true;
}

bool ATEInterface::sendStressWarning(float max_stress, float yield_threshold,
                                   int critical_count, const std::string& location) {
    if (!checkConnection()) return false;
    
    std::stringstream warning;
    warning << "WARNING: Stress exceeds " << (max_stress / yield_threshold) * 100.0f 
            << "% of yield threshold at " << location 
            << " (" << critical_count << " critical points detected)";
    
    std::cout << warning.str() << std::endl;
    
    return sendCommand("LOG " + warning.str());
}

void ATEInterface::disconnect() {
    if (connected) {
        std::cout << "Disconnecting from ATE system" << std::endl;
        
        // In a real implementation, would close the socket
        socket_fd = -1;
        connected = false;
    }
}

bool ATEInterface::checkConnection() {
    if (!connected) {
        std::cerr << "Error: Not connected to ATE system" << std::endl;
        return false;
    }
    return true;
} 