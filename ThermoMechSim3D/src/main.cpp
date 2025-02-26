#include <iostream>
#include <chrono>
#include <string>
#include "ThermoMechSimManager.h"

void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " [options]\n";
    std::cout << "Options:\n";
    std::cout << "  -c, --config <file>   Specify configuration file (default: config.json)\n";
    std::cout << "  -o, --output <dir>    Specify output directory (default: from config)\n";
    std::cout << "  -s, --steps <num>     Override number of simulation steps\n";
    std::cout << "  -d, --device <id>     Specify CUDA device ID\n";
    std::cout << "  -h, --help            Display this help message\n";
}

int main(int argc, char** argv) {
    std::string configFile = "config.json";
    std::string outputDir = "";
    int steps = -1;
    int deviceId = -1;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "-c" || arg == "--config") {
            if (i + 1 < argc) {
                configFile = argv[++i];
            } else {
                std::cerr << "Error: --config requires a file path\n";
                printUsage(argv[0]);
                return 1;
            }
        } else if (arg == "-o" || arg == "--output") {
            if (i + 1 < argc) {
                outputDir = argv[++i];
            } else {
                std::cerr << "Error: --output requires a directory path\n";
                printUsage(argv[0]);
                return 1;
            }
        } else if (arg == "-s" || arg == "--steps") {
            if (i + 1 < argc) {
                steps = std::stoi(argv[++i]);
            } else {
                std::cerr << "Error: --steps requires a number\n";
                printUsage(argv[0]);
                return 1;
            }
        } else if (arg == "-d" || arg == "--device") {
            if (i + 1 < argc) {
                deviceId = std::stoi(argv[++i]);
            } else {
                std::cerr << "Error: --device requires a device ID\n";
                printUsage(argv[0]);
                return 1;
            }
        } else if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            printUsage(argv[0]);
            return 1;
        }
    }
    
    std::cout << "ThermoMechSim3D - CUDA-Accelerated Thermal-Mechanical Stress Simulation\n";
    std::cout << "===============================================================\n";
    
    // Create and initialize the simulation manager
    ThermoMechSimManager simManager;
    
    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Initialize simulation
    if (!simManager.initialize(configFile)) {
        std::cerr << "Failed to initialize simulation\n";
        return 1;
    }
    
    // Apply command-line overrides if provided
    if (!outputDir.empty()) {
        simManager.setOutputDirectory(outputDir);
    }
    
    if (steps > 0) {
        simManager.setNumSteps(steps);
    }
    
    if (deviceId >= 0) {
        simManager.setDeviceId(deviceId);
    }
    
    // Print simulation information
    simManager.printSimulationInfo();
    
    // Run simulation
    if (!simManager.run()) {
        std::cerr << "Simulation failed\n";
        return 1;
    }
    
    // Generate report
    if (!simManager.generateReport("simulation_report.txt")) {
        std::cerr << "Failed to generate simulation report\n";
        return 1;
    }
    
    // End timing
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    
    std::cout << "------------------ Simulation completed ------------------\n";
    std::cout << "Elapsed time: " << elapsed.count() << " seconds\n";
    std::cout << "Maximum temperature: " << simManager.getMaxTemperature() << " K\n";
    std::cout << "Maximum stress: " << simManager.getMaxStress() / 1e6 << " MPa\n";
    std::cout << "Critical regions: " << simManager.getNumCriticalRegions() << "\n";
    std::cout << "==========================================================\n";
    
    return 0;
} 