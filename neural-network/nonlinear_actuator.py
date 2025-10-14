# Simple example to demonstrate compensation of a nonlinear actuator

# Intended as an educational exercise, this script trains a neural
# network to modify actuator inputs to account for nonlinearity

# %%

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf

from moku.nn import LinnModel, save_linn

# %%
# ---
# Step 1: Generate the actuator response (we can also use this as our training data)
# ---


def actuator_response(x):
    return erf(2 * x)


nPts = 2**10  # Investigate how model fit depends on amount of data supplied and training ratio
input = np.linspace(-1, 1, nPts)
output = actuator_response(input)
# plt.figure()
# plt.plot(input, output)
# plt.xlabel('Input')
# plt.ylabel('Output')
# plt.title('Actuator response')

# Observe actuator output for a typical input
phase = np.linspace(0, 3 * 2 * np.pi, nPts)
desired = np.sin(phase)
achieved = actuator_response(desired)
# plt.figure()
# plt.plot(phase, desired, label='Desired')
# plt.plot(phase, achieved, label='Achieved')
# plt.xlabel('Time')
# plt.ylabel('Output')
# plt.title('Actuator output')
# plt.legend()


# %%
# ---
# Step 2: Train a neural network to compensate for the nonlinearity
# ---

# Reshape the inputs, ready for training. Each `data point' is one input and one output
input.shape = [nPts, 1]
output.shape = [nPts, 1]

# Build the neural network model. Skip the I/O scaling as the data are already pretty good for training,
# Note that if you add scaling here, you'll need to apply the same scaling in the Moku Neural Network instrument at
# runtime.
# Note the model inputs are the actuator outputs and vice versa
# We give the NN the output we want and it generates the appropriate input to apply to the actuator
linn_model = LinnModel()
linn_model.set_training_data(output, input, scale=False)

# Try different models
# model_definition = [ (1, 'linear')] # The most basic linear model. Can't cope with nonlinearity
model_definition = [
    (16, 'relu'),
    (16, 'relu'),
    (16, 'relu'),
    (1, 'linear'),
]  # ReLU activation works well in this example
# model_definition = [ (100, 'relu'), (100, 'relu'), (100, 'relu'), (100, 'relu'), (100, 'relu')] # Biggest Moku can fit, can overfit!

linn_model.construct_model(model_definition)

# # %%
# # Train the model. This simple model converges pretty quickly so an early stopping config terminates training much more quickly
history = linn_model.fit_model(epochs=500, validation_split=0.1, es_config={'patience': 16})

# plot the losses
plt.figure()
plt.semilogy(history.history['loss'])
plt.semilogy(history.history['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.xlabel('Epochs')
plt.title('Loss functions')


# Plot the NN model of the actuator
# Given the output, model should provide the input that will give rise to it
nn_out = linn_model.predict(output)

plt.figure()
plt.plot(input, output, label='True response')
plt.plot(nn_out, output, '--', label='Modelled response')
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('Actuator response')
plt.legend()

# %%
# Investigate linearisation
nn_out2 = linn_model.predict(input)

fig, axs = plt.subplots(2)
fig.suptitle('Linearisation')
axs[0].plot(input, input, label='Ideal actuator')
axs[0].plot(input, actuator_response(nn_out2), '--', label='Linearised')
axs[0].legend()
axs[1].plot(input - actuator_response(nn_out2), label='Ideal - Linearised')
axs[1].legend()

# %%
# Plot the output for a typical waveform
desired_reshaped = desired.copy()
desired_reshaped.shape = [nPts, 1]
nn_out3 = linn_model.predict(desired_reshaped)

fig2, axs2 = plt.subplots(2)
axs2[0].plot(phase, desired, label='Desired')
axs2[0].plot(phase, achieved, label='W/o correction')
axs2[0].plot(phase, actuator_response(nn_out3), '--', label='W/ NN correction')
axs2[0].set_ylabel('Output')
fig2.suptitle('Actuator output')
axs2[0].legend()

axs2[1].plot(
    phase, desired_reshaped - actuator_response(nn_out3), label='Desired - W/ NN correction'
)
axs2[1].set_xlabel('Time')
axs2[1].legend()
plt.show()

# %%
# ---
# Step 3: Save the model to disk for use in the Moku Neural Network instrument
# ---

save_linn(linn_model, input_channels=1, output_channels=1, file_name='nonlinear_actuator.linn')

# %%
