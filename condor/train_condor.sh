#!/bin/bash

# Load necessary modules for GPU and CPU parallelization
module load gcc
module load cuda
module load cudnn

# Activate the Python environment with required dependencies
source activate my_cVAE_env

# Train the cVAE model
python train.py
