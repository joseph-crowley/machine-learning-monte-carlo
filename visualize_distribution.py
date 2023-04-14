import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import os
from cVAE import cVAE  

def generate_samples(model, num_samples):
    latent_samples = torch.randn(num_samples, model.latent_dim)
    with torch.no_grad():
        generated_data = model.decoder(latent_samples)
    return generated_data.numpy()

def plot_3d_scatter(data, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(data[:, 0], data[:, 1], data[:, 2], marker='o')
    ax.set_title(title)

    # style the plot for better visualization
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # set the perspective from above the torus along the z-axis
    ax.view_init(90, 0)

    # make it professional
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('white')
    ax.yaxis.pane.set_edgecolor('white')
    ax.zaxis.pane.set_edgecolor('white')
    ax.xaxis._axinfo['grid'] = {'color': (1, 1, 1, 0)}
    ax.yaxis._axinfo['grid'] = {'color': (1, 1, 1, 0)}
    ax.zaxis._axinfo['grid'] = {'color': (1, 1, 1, 0)}


    # save the figure
    fig.savefig('{}.png'.format(title))

    #plt.tight_layout()
    #plt.show()

def generate_torus_data():
    # Load trained cVAE model
    model_save_path = 'trained_cVAE'
    cVAE_model_loaded = torch.load(os.path.join(model_save_path, 'trained_cVAE.pt'))

    # Generate samples from the learned distribution
    num_samples = int(1e6)
    return generate_samples(cVAE_model_loaded, num_samples)

def get_torus_data(file_path='data/toroid_dataset.csv'):
    dataset = pd.read_csv(file_path)
    return dataset.values

# Plot the original torus data
file_path = 'data/toroid_dataset.csv'
torus_data = get_torus_data(file_path=file_path)
plot_3d_scatter(torus_data, title='Original 3D Toroidal Distribution Visualization')


# Plot the generated samples
generated_data = generate_torus_data()
plot_3d_scatter(generated_data, title='Learned 3D Toroidal Distribution Visualization')

