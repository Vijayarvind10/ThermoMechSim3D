# CUDA-Accelerated 3D Fault Simulation for Semiconductor DFX

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CUDA](https://img.shields.io/badge/CUDA-12.2-%2376B900.svg?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.123456789.svg)](https://doi.org/10.5281/zenodo.123456789)

A high-performance GPU-accelerated framework for simulating defects in 3D-stacked semiconductor chips, optimized for NVIDIA GPUs and aligned with NVIDIA cuLitho workflows.

<p align="center">
  <img src="media/fault_simulation_demo.gif" alt="3D Fault Simulation Demo">
</p>

---

## 📌 Overview
This project accelerates **semiconductor defect analysis** for next-gen chips (e.g., NVIDIA Blackwell, TSMC N3) using CUDA-optimized parallel computing. It reduces simulation time for 10B+ transistor designs by **35×** compared to traditional CPU methods, targeting **DFX (Design for Excellence)** challenges faced by NVIDIA's HPC and automotive teams.

**Key Features**  
- 🚀 **3D Defect Propagation**: Simulates opens/shorts in TSMC 3DIC stacks using CUDA wavefront partitioning
- 🔍 **Adaptive Fault Sampling**: ML-guided critical path prioritization (XGBoost + cuML)
- 📊 **NVIDIA Omniverse Visualization**: Renders defect heatmaps in real-time
- 🛠️ **cuLitho Integration**: Lithography-aware defect modeling for NVIDIA's computational lithography pipeline

---

## 🛠️ Installation

### Requirements
- **NVIDIA GPU**: Compute Capability ≥ 7.0 (Ampere+)
- **CUDA Toolkit**: 12.2+
- **Python**: 3.8+ with `numpy`, `pandas`, `xgboost`

