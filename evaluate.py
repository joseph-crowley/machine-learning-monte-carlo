import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from cVAE import cVAE, cVAE_loss  # Assuming you have saved the PyTorch implementation of cVAE in cVAE.py

import torch

# Load dataset from CSV file
def load_dataset(file_path):
    dataset = pd.read_csv(file_path)
    return dataset.values

# Evaluate cVAE model
def evaluate_cVAE_model(model, x_test):
    x_test = torch.tensor(x_test, dtype=torch.float32)
    with torch.no_grad():
        reconstructed_data, z_mean, z_log_var = model(x_test)
    mse = mean_squared_error(x_test, reconstructed_data.numpy())
    return mse

# Load dataset
file_path = 'data/toroid_dataset.csv'
dataset = load_dataset(file_path)

# Split dataset into training, validation, and test sets
x_train_val, x_test = train_test_split(dataset, test_size=0.2, random_state=42)
x_train, x_val = train_test_split(x_train_val, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2

# Load trained cVAE model
model_save_path = 'trained_cVAE'
#cVAE_encoder_loaded = torch.load(os.path.join(model_save_path, 'encoder.pth'))
#cVAE_decoder_loaded = torch.load(os.path.join(model_save_path, 'decoder.pth'))
cVAE_model_loaded = torch.load(os.path.join(model_save_path, 'trained_cVAE.pt'))

# Evaluate the cVAE model
mse = evaluate_cVAE_model(cVAE_model_loaded, x_test)
print('Mean Squared Error: {:.4f}'.format(mse))

torch.save(cVAE_model_loaded, 'trained_cVAE.pt')