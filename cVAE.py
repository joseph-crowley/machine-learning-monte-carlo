import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

# Custom conditional Variational Autoencoder (cVAE) class
class cVAE(nn.Module):
    def __init__(self, hidden_dim, latent_dim, input_dim=3, encoder=None, decoder=None):
        super(cVAE, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        
        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = self.build_encoder()

        if decoder is not None:
            self.decoder = decoder
        else:
            self.decoder = self.build_decoder()
    
    # Encoder network
    def build_encoder(self):
        return nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
    
    # Decoder network
    def build_decoder(self):
        return nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.input_dim),
        )
    
    # Sampling from latent space
    def sample_from_latent(self, z_mean, z_log_var):
        epsilon = torch.randn_like(z_mean)
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon
    
    # Full forward pass
    def forward(self, inputs):
        x = self.encoder(inputs)
        z_mean, z_log_var = x.chunk(2, dim=-1)
        z = self.sample_from_latent(z_mean, z_log_var)
        reconstructed = self.decoder(z)
        return reconstructed, z_mean, z_log_var

# Custom cVAE loss function
def cVAE_loss(y_true, y_pred, z_mean, z_log_var):
    reconstruction_loss = F.mse_loss(y_pred, y_true, reduction='none').sum(dim=-1)
    kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean**2 - z_log_var.exp(), dim=-1)
    return (reconstruction_loss + kl_loss).mean()

# Hyperparameters
hidden_dim = 128
latent_dim = 32

# Create and compile conditional Variational Autoencoder
cVAE_model = cVAE(hidden_dim, latent_dim)

# Optimizer
optimizer = Adam(cVAE_model.parameters())

# Summary of the cVAE architecture
print("Encoder Model")
print(cVAE_model.encoder)
print("\nDecoder Model")
print(cVAE_model.decoder)
print("\nComplete cVAE Model")
print(cVAE_model)