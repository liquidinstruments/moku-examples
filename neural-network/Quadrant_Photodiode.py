# Simple example of a use case for the Moku Neural Network instrument.
# This script generates a Quadrant Photodiode function with some noise and distortion, and then trains a neural network
# to predict the beam position from the QPD values. A few network types are listed to experiment with, the recommended
# one has two hidden layers of 16 neurons each, and uses the tanh activation function.
#
# The model is trained on a grid of points in the beam position space, and the error between the actual and predicted
# beam positions is plotted compared to the naive calculation. The naive calculation is also plotted for comparison.
#
# The model is then saved to disk for use with the Moku Neural Network instrument.

# %%
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import dblquad

from moku.nn import LinnModel, save_linn

# %%
# ---
# Step 1: Simulate the QPD function to build input and output data for training
# ---


# Gaussian spot function with elliptical symmetry and variable intensity
def spot_func(x, y, X, Y, spot=0.5, intensity=1, sym=1):
    return np.exp(-((x - X) ** 2 * sym + (y - Y) ** 2 / sym) / spot) * intensity


# Simple QPD function, integrating the spot function over each quadrant. Assumes square detectors
def position_to_quadrants(X, Y, spot=0.5, intensity=1, sym=1):
    # Integrate over each quadrant using numerical integration
    def integrand(x, y):
        return spot_func(x, y, X, Y, spot, intensity, sym)

    q1_val = dblquad(integrand, 0, 1, -1, 0)[0]
    q2_val = dblquad(integrand, 0, 1, 0, 1)[0]
    q3_val = dblquad(integrand, -1, 0, -1, 0)[0]
    q4_val = dblquad(integrand, -1, 0, 0, 1)[0]

    return np.array((q1_val, q2_val, q3_val, q4_val))


# Add some distortion and noise to the QPD function. Compute some static biases and gain errors
# for each quadrant diode once per run, with some random noise added for each point
biases = np.random.normal(scale=0.01, size=4)
gains = np.random.normal(loc=1, scale=0.1, size=4)
sym = np.random.normal(loc=1, scale=0.3)


def position_to_quadrants_noisy(X, Y):
    spot = np.random.normal(loc=0.5, scale=0.1)
    intensity = np.random.normal(loc=1, scale=0.1)

    qs = position_to_quadrants(X, Y, spot, intensity, sym)

    # Apply systematic errors
    qs = qs * gains + biases

    # Gain and bias random errors
    qs += np.random.normal(scale=0.01, size=4)
    qs *= np.random.normal(loc=1, scale=0.005, size=4)
    return qs


# Simple naive point calculation from the quadrant values as a baseline
def naive_point(q1, q2, q3, q4):
    x = q2 - q1 + q4 - q3
    y = q1 + q2 - q3 - q4

    i = q1 + q2 + q3 + q4
    return (x / i, y / i)


# %%
# Generate a grid of points to evaluate the QPD function over. Plot the spot function
# at the origin as an example and sanity check
x = np.linspace(-1, 1, 100)
X, Y = np.meshgrid(x, x)
plt.imshow(spot_func(X, Y, 0, 0), extent=(-1, 1, -1, 1))

# %%
# Generate the QPD function values for each point in the grid. Cache the results to disk. Set
# regenerate to True to recompute the QPD values when you change the simulation function or mesh
regenerate = True
update_cache = True
cache_file = 'quadrant_data.h5'
if not os.path.exists(cache_file) or regenerate:
    qg = np.vectorize(position_to_quadrants_noisy, signature='(),()->(4)')(X, Y)

    if update_cache:
        with h5py.File(cache_file, 'w') as hf:
            hf.create_dataset('qg', data=qg)


with h5py.File(cache_file, 'r') as hf:
    qg = hf['qg'][:]
# %%
# Plot the QPD function values for each quadrant.
fig, ax = plt.subplots(1, 4, subplot_kw={'projection': '3d'})
for i in range(4):
    ax[i].plot_surface(X, Y, qg[:, :, i], cmap='viridis')
    ax[i].set_xlabel('X')
    ax[i].set_ylabel('Y')
# %%
# Compute the naive point calculation from the QPD values and compare to the actual points. Plot the error
# between the naive and actual points as a function of (actual) beam position
naive_points = np.apply_along_axis(lambda x: naive_point(*x), 2, qg)
actual_points = np.stack((X, Y), axis=-1)
error = np.linalg.norm(naive_points - actual_points, axis=-1)

plt.imshow(error, extent=(-1, 1, -1, 1), vmax=0.5)
plt.colorbar()

# %%
# ---
# Step 2: Train a neural network to predict the beam position from the QPD values
# ---

# Reshape the evaluation mesh to a list, ready for training
locations = actual_points.reshape(-1, 2)
q_values = qg.reshape(-1, 4)

# Build the neural network model. Skip the I/O scaling as the data are already pretty good for training,
# though QPD powers (input values) are strictly positive so input scaling might be beneficial depending on the model.
# Note that if you add scaling here, you'll need to apply the same scaling in the Moku Neural Network instrument at
# runtime.
linn_model = LinnModel()
linn_model.set_training_data(q_values, locations, scale=False)

# The input layer is inferred, so only the hidden and output layers need to be defined.
# The output layer must have dimension 2 to match the output point (X, Y) and must have an
# activation function that can go negative, i.e. linear, tanh, or softsign

# model_definition = [ (4, 'relu'), (4, 'relu'), (2, 'linear')] # ReLU activation not great on signal reconstruction
# model_definition = [ (4, 'tanh'), (4, 'tanh'), (2, 'linear')] # tanh has a range of +-1 which is more "voltage-like"
model_definition = [
    (16, 'tanh'),
    (16, 'tanh'),
    (2, 'linear'),
]  # A few more degrees of freedom to play around with starts to give good results
# model_definition = [ (100, 'tanh'), (100, 'tanh'), (100, 'tanh'), (100, 'tanh'), (2, 'linear')] # Biggest Moku can fit, can overfit!
linn_model.construct_model(model_definition)

# %%
# Train the model. This simple model converges pretty quickly so an early stopping config terminates training much more quickly
history = linn_model.fit_model(epochs=500, validation_split=0.1, es_config={'patience': 10})

# plot the losses
plt.semilogy(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.xlabel('Epochs')
plt.show()

# %%
# Plot the error between the actual and predicted points to compare to the naive calculation above
nn_points = linn_model.predict(q_values)
nn_error = np.linalg.norm(nn_points - locations, axis=-1)
plt.imshow(nn_error.reshape(*X.shape), extent=(-1, 1, -1, 1), vmax=0.5)
plt.colorbar()

# %%
# ---
# Step 3: Save the model to disk for use in the Moku Neural Network instrument
# ---
save_linn(linn_model, input_channels=4, output_channels=2, file_name='qpd.linn')

# %%
