# Toroidal cVAE

This repository contains code for training, evaluating, and visualizing cVAE models on Toroidal 3D data.

## Prerequisites

- Python 3.11
- Conda & Pip
- CUDA and cuDNN (if using GPU)

## Installation

1. Clone this repository.
2. Create and activate a conda environment using the `environment.yml` file:
   ```
   conda env create -f condor/environment.yml
   conda activate my_cVAE_env
   ```
3. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

- Train cVAE model:
  ```
  python train.py
  ```

- Evaluate the trained model:
  ```
  python evaluate.py
  ```

- Visualize the learned distribution:
  ```
  python visualize_distribution.py
  ```

## Running on Condor

- Adjust the `train_condor.sub` file according to your desired resources.

- Submit the job:
  ```
  condor_submit condor/train_condor.sub
  ```