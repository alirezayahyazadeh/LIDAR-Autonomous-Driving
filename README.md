# 3D Object Detection Using NuScenes and SECOND Framework

## Overview

This repository contains scripts for 3D object detection using the NuScenes dataset and the SECOND deep learning framework. The project is designed to preprocess data, train a 3D object detection model(pedestrian), evaluate its performance, and visualize the results. It utilizes PyTorch, NuScenes devkit, and SECOND for efficient and scalable 3D object detection workflows.

## Features

- **NuScenes Dataset Preprocessing**: Converts NuScenes data into a format compatible with the SECOND framework.
- **Training with SECOND**: Uses the SECOND framework for training a high-performance 3D object detection model.
- **Model Evaluation**: Generates performance metrics and logs results.
- **Result Visualization**: Visualizes predictions on 3D point clouds and bounding boxes using NuScenes visualization tools.

## File Structure

### Core Scripts

1. **t1.py**  
   - Preprocesses the NuScenes dataset for the SECOND framework.
   - Trains a 3D object detection model.
   - Evaluates the trained model and visualizes the results.

### Additional Components
- **NuScenes Dataset**: Preprocessed data is stored in `./data/processed`.
- **Model Configurations**: SECOND framework configurations are stored in `./configs/second_config.yaml`.
- **Checkpoints and Results**: Saved models and evaluation metrics are in `./checkpoints`.

## Requirements

- Python 3.8+
- Libraries:
  - `torch`
  - `nuscenes-devkit`
  - `SECOND`
  - `matplotlib`
  - `numpy`

## Installation

1. Clone the repository:

   ```
   git clone https://github.com/alirezayahyazadeh/LIDAR-Autonomous-Driving
   cd yourrepository

Install the required dependencies:


pip install -r requirements.txt
Ensure requirements.txt includes:

torch
nuscenes-devkit
matplotlib
numpy
Usage

_Step 1: Preprocess the NuScenes Dataset
Download the NuScenes dataset and place it in ./data/nuscenes. Then run:


python t1.py
_Step 2: Train the Model
Ensure second_config.yaml is properly configured, and execute the training step within t1.py.

_Step 3: Evaluate the Model
After training, the evaluation step in t1.py will generate performance metrics.

_Step 4: Visualize Results
The script includes visualization tools to review detection results on 3D point clouds.

Key Commands
Start Preprocessing: python t1.py
Stop Visualization: Press ESC.
Contribution
Contributions are welcome! Fork the repository, create a branch for your changes, and submit a pull request. Let us improve the future of 3D object detection together!
