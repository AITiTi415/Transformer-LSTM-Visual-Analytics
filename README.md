# Group-Consensus Visual Analytics for Multimodal Driving Cognition Assessment

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-ee4c2c.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

> **Citation Note:** This repository contains the official codebase, benchmark protocols, pretrained model checkpoints, and visual analytics terminal for the manuscript ***"Group-Consensus Visual Analytics for Multimodal Driving Cognition Assessment in High-Risk Virtual Scenarios"***, currently submitted to ***The Visual Computer***. 
> 
> If you find our codebase, evaluation protocols, or visual terminal useful in your research, please consider citing our paper upon publication.

---

##  Overview

This repository provides a comprehensive multimodal visual analytics framework for assessing driving cognition from non-intrusive simulated-driving data. By integrating 5D temporal physical/visual features (steering, throttle, braking, SGE, and GTE) with spatial scene priors (1D-MASPP), the framework decodes multimodal behaviors into eight interpretable cognitive dimensions using a spatiotemporal dual-branch decoupling network and a self-supervised pseudo-label alignment mechanism.

##  Repository Structure

* `models.py` / `xiaorong_models.py` / `BaselineModels.py`: Core neural network architectures, including the proposed dual-branch framework, ablation variants, and state-of-the-art baselines.
* `main.py`: Main execution script for end-to-end training and task-level reliability evaluation.
* `dataloader.py`: Subject-level data loading and sequence batching pipeline.
* `build_priors.py`: Script to construct the discrete spatial prior semantic matrix for virtual scenarios.
* `run_Baseline.py` / `run xiaorong.py`: Benchmark execution protocols for comparative and structural ablation studies.
* `plot_explainability.py`: Scripts for rendering high-resolution academic figures (e.g., Attention Heatmap, SHAP Matrix).
* `Visual.py`: The interactive Streamlit-based Visual Analytics Terminal for expert decision support.
* `Data Process/`: Contains the raw data preprocessing pipeline and configuration scripts.
* `data/`, `checkpoints/`, `outputs/`: Directories storing the extracted feature matrices, pretrained model weights, and generated evaluation figures.

##  Environment Setup

To ensure full reproducibility as recommended by the evaluation protocols, please clone the repository and install the required dependencies using the provided environment file:

```bash
git clone [https://github.com/AITiTi415/Transformer-LSTM-Visual-Analysis.git](https://github.com/AITiTi415/Transformer-LSTM-Visual-Analysis.git)
cd Transformer-LSTM-analysis

# Install dependencies
pip install -r requirements.txt
