#ifndef ATE_INTERFACE_H
#define ATE_INTERFACE_H

#include <string>
#include <vector>

// Interface for Automated Test Equipment communication
class ATEInterface {
public:
    ATEInterface();
    ~ATEInterface();
    
    // Initialize connection to ATE
    bool initialize(const std::string& ate_address, int port);
    
    // Send command to ATE
    bool sendCommand(const std::string& command);
    
    // Send bin information and stress test results
    bool sendBinResult(int bin_number, const std::string& failure_reason);
    
    // Send detailed stress map
    bool sendStressData(const std::vector<float>& stress_map, 
                       int dim_x, int dim_y, int dim_z);
    
    // Send warning about critical stress
    bool sendStressWarning(float max_stress, float yield_threshold,
                         int critical_count, const std::string& location);
    
    // Close connection
    void disconnect();
    
private:
    // Connection state
    bool connected;
    std::string ate_address;
    int port;
    
    // Mock socket for ATE communication
    int socket_fd;
    
    // Utility functions
    bool checkConnection();
    std::string formatStressData(const std::vector<float>& stress_map,
                                int dim_x, int dim_y, int dim_z);
};

#endif // ATE_INTERFACE_H 