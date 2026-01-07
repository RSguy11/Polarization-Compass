# Polarization-Compass

Bio-inspired polarization compass for underwater navigation using machine learning models to predict solar azimuth from polarization data.

## Overview

As the world has evolved, humanity has ventured to different regions of the world, both above the clouds and below the sea. Whether the nature of these ventures is explorative, militaristic, or for pleasure, these domains produce unique technological challenges that drive innovation. 

One such challenge is navigation in underwater environments. This repository stores all AI models, noise reduction software and software components related to the construction of an underwater polarization-based compass.

## Project Structure

### Machine Learning Models
- **L2_Linear_reg/**: L2 (Ridge) linear regression baseline model
- **SVR_reg/**: Support Vector Regression with RBF kernel  
- **Random_Forest_reg/**: Random Forest regressor with ensemble regularization
- **Training_loops/**: Complete training pipelines for each model

### Data Processing
- **Preprocessing/**: Multi-stage polarization image processing pipeline
  - **stage1/**: Denoising and interpolation
  - **stage2/**: Stokes parameter computation and DoLP/AoLP extraction
  - **stage3/**: Advanced polarization restoration
- **solar_azimuth_generator.py**: Ground truth azimuth label generation

### Utilities
- **model_comparison.py**: Comprehensive model evaluation and comparison
- **run_all_models.py**: Unified training pipeline for all models
- **ml_data_integration.py**: Data pipeline integration utilities

## Quick Start

1. **Test Baseline Models:**
   ```bash
   python test_l2_implementation.py
   python run_all_models.py
   ```

2. **Generate Solar Azimuth Labels:**
   ```bash
   python solar_azimuth_generator.py
   ```

3. **Run Model Comparison:**
   ```bash
   python model_comparison.py
   ```

## Requirements

Target performance (per project blueprint):
- **MAE < 5°** for solar azimuth prediction
- **RMSE ≤ 10%**
- Cross-validation with k=5 folds
- 20k supervised dataset (10k original + 10k augmented)
