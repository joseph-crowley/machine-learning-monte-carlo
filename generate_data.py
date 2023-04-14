import numpy as np
import pandas as pd
import random

def toroidal_coordinates(r, R, theta, phi):
    """
    Calculate the toroidal coordinates (x, y, z) given the minor radius r,
    major radius R, and angles theta and phi.

    :param r: minor radius of the toroid
    :param R: major radius of the toroid
    :param theta: angle in radians in the range [0, 2 * pi)
    :param phi: angle in radians in the range [0, 2 * pi)
    :return: tuple (x, y, z) representing the toroidal 3D coordinates
    """
    x = (R + r * np.cos(phi)) * np.cos(theta)
    y = (R + r * np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)
    return x, y, z

def generate_toroid_dataset(r, R, n_points):
    """
    Generate a toroid dataset using toroidal coordinates.

    :param r: minor radius of the toroid
    :param R: major radius of the toroid
    :param n_points: number of points in the dataset
    :return: a numpy array of shape (n_points, 3) containing x, y, z coordinates of the toroid points
    """
    toroid_points = []
    for i in range(n_points):
        theta = 2 * np.pi * random.random()
        phi = 2 * np.pi * random.random()
        x, y, z = toroidal_coordinates(r, R, theta, phi)
        toroid_points.append([x, y, z])
    return np.array(toroid_points)

def save_toroid_dataset(data, filename):
    """
    Save the toroid dataset to a CSV file.

    :param data: numpy array containing x, y, z coordinates of the toroid points
    :param filename: the name of the output CSV file
    """
    df = pd.DataFrame(data, columns=["x", "y", "z"])
    df.to_csv(filename, index=False)


# Parameters for the toroid, dataset size and output file

# create the toroid with a big hole in the middle
minor_radius = 0.5
major_radius = 1.0
num_points = int(1e8)
output_file = "data/toroid_dataset.csv"

# Generate the dataset
toroid_data = generate_toroid_dataset(minor_radius, major_radius, num_points)

# Save the dataset to a CSV file
save_toroid_dataset(toroid_data, output_file)
