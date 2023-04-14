import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from cVAE import cVAE

# Load dataset from CSV file
def load_dataset(file_path):
    dataset = pd.read_csv(file_path)
    return dataset.values

class NumpyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Train cVAE model
def train_cVAE_model(model, dataloader, val_dataloader, epochs, learning_rate, patience):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, verbose=True)
    loss_fn = torch.nn.MSELoss()
    history = []
    best_val_loss = None
    no_improvement_count = 0

    for epoch in range(epochs):
        model.train()
        for x in dataloader:
            x = x.to(torch.float32)
            optimizer.zero_grad()
            recon_x, _, _ = model(x)
            loss = loss_fn(recon_x, x)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x in val_dataloader:
                x = x.to(torch.float32)
                recon_x, _, _ = model(x)
                val_loss += loss_fn(recon_x, x).item()
        val_loss /= len(val_dataloader)

        print(f'Epoch: {epoch+1}/{epochs}, Loss: {loss.item()}, Val Loss: {val_loss}')
        history.append({'loss': loss.item(), 'val_loss': val_loss})
        
        scheduler.step(val_loss)

        if best_val_loss is None or val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if no_improvement_count >= patience:
            print("Early stopping triggered")
            break

    return history

# Load dataset
file_path = 'data/toroid_dataset.csv'
dataset = load_dataset(file_path)

# Split dataset into training and validation sets
x_train, x_val = train_test_split(dataset, test_size=0.2, random_state=42)

# Hyperparameters
batch_size = 64
epochs = 150
learning_rate = 1e-3
patience = 20

train_dataset = NumpyDataset(x_train)
val_dataset = NumpyDataset(x_val)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize the cVAE model with appropriate hyperparameters
cVAE_model = cVAE(latent_dim=64, hidden_dim=128, input_dim=3)

# Train the cVAE model
training_history = train_cVAE_model(cVAE_model, train_dataloader, val_dataloader, epochs, learning_rate, patience)

# Save the cVAE model
model_save_path = 'trained_cVAE'
os.makedirs(model_save_path, exist_ok=True)
torch.save(cVAE_model.encoder.state_dict(), os.path.join(model_save_path, 'encoder.pth'))
torch.save(cVAE_model.decoder.state_dict(), os.path.join(model_save_path, 'decoder.pth'))
torch.save(cVAE_model, os.path.join(model_save_path, 'trained_cVAE.pt'))

print('Model training is complete and the models are saved.')
