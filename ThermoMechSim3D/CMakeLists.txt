cmake_minimum_required(VERSION 3.18)
project(ThermoMechSim3D CUDA CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Set CUDA flags
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} 
    -O3 
    -arch=sm_70 
    --use_fast_math 
    -Xcompiler 
    -Wall,-Wextra)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Source files
set(SOURCES
    src/io/input_parser.cpp
    src/io/material_db.cpp
    src/visualization/vtk_writer.cpp
    src/ThermoMechSimManager.cpp
    src/main.cpp
)

# CUDA source files
set(CUDA_SOURCES
    src/cuda/thermal_stress.cu
    src/cuda/utils.cu
)

# Generate CUDA object files
cuda_compile(CUDA_OBJECTS ${CUDA_SOURCES})

# Create executable
add_executable(ThermoMechSim3D ${SOURCES} ${CUDA_OBJECTS})

# Link libraries
target_link_libraries(ThermoMechSim3D ${CUDA_LIBRARIES})

# Copy example files to build directory
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/examples/ 
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/examples/)

# Installation
install(TARGETS ThermoMechSim3D DESTINATION bin)
install(DIRECTORY examples/ DESTINATION share/ThermoMechSim3D/examples) 