# Anomaly Detection example

<example-actions directory="neural-network" filename="Anomaly_detection.ipynb"/>

An autoencoder is a type of neural network that learns to reconstruct its input, making it useful for anomaly detection in signals. If a given input contains an unusual pattern that the network hasn't learned well, the reconstruction error will be high for that part, identifying the anomaly.

In this example, we will train an autoencoder to model a known periodic signal, which can flag anomalies via a large reconstruction error. We will use a Moku device to capture the signal data, then train a neural network model that can later be deployed on the Moku for real-time anomaly detection.

# Prepare the environment

Optional: if running on a fresh environment like Google Colab, install or upgrade the Moku API library (with Neural Network support) before proceeding.

```python
# Uncomment and run this if Moku library is not installed
# !pip install -U "moku[neuralnetwork]"
```
First, we import the necessary Python libraries.

```python
import csv
import os
import time
import numpy as np
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm
```

Next, we import the Moku instrument classes we need, as well as the neural network model utilities.

```python
from moku.instruments import MultiInstrument, Oscilloscope

try:
    from moku.nn import LinnModel, save_linn
except ImportError:
    print("Moku library is not installed. If running in an online notebook, install it with the pip command above.")
    raise
```

We'll define some helper functions for later use, particularly for formatting and data I/O.

```python
def format_sampling_rate(rate):
    """Format a sampling rate in Hz to kHz or MHz if applicable."""
    if rate >= 1e6:
        return f"{rate/1e6:.2f} MHz"
    elif rate >= 1e3:
        return f"{rate/1e3:.2f} kHz"
    else:
        return f"{rate:.2f} Hz"

def save_data_to_csv(data_points, window_size, filename):
    """Save 1D data points into CSV file, splitting into fixed-size windows per row."""
    # Ensure all values are floats
    data_points = [float(x) for x in data_points]

    # Split data into consecutive windows of length window_size
    windows = []
    num_points = len(data_points)
    i = 0
    while i + window_size <= num_points:
        windows.append(data_points[i:i + window_size])
        i += window_size
    # Handle the last partial window (pad or adjust start to full length)
    if i < num_points:
        start = max(0, num_points - window_size)
        windows.append(data_points[start:num_points])

    # Determine next ID for data frames by counting existing rows in file
    file_exists = os.path.exists(filename)
    next_id = 0
    if file_exists:
        with open(filename, 'r', newline='') as file:
            reader = csv.reader(file)
            next(reader, None)  # skip header
            next_id = sum(1 for _ in reader)

    # Append new data windows to the CSV
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['id', 'data'])  # write header if new file
        for w in windows:
            writer.writerow([next_id, ';'.join(f"{x:.6f}" for x in w)])
            next_id += 1

def load_data_from_csv(filename):
    """Load data windows from the CSV file back into a list of lists of float values."""
    data_windows = []
    with open(filename, 'r', newline='') as file:
        reader = csv.reader(file)
        header = next(reader, None)  # skip header if present
        for row in reader:
            if len(row) < 2:
                continue  # skip malformed rows
            # Each row's second column is a ';'-separated list of values
            values = [float(x) for x in row[1].split(';')]
            data_windows.append(values)
    return data_windows
```

We can see the available Moku devices, using the Moku CLI through the command line as a quick check.

```python
! mokucli list
```
```
| Name | Serial | HW | FW | IP |
|------|--------|----|----|----|
|      |        |    |    |    |
```

The above command should output a list of connected Moku devices with their name, serial, IP, etc. Ensure your Moku device is listed and reachable.

# Set up the Oscilloscope for data collection

First, set up the folder structure for organizaing output data and define the filename for captured data frames

```python
# Define output folder and base filename
output_folder = "AD_dataset/"
output_filename = "data_training"

# Generate a unique filename with timestamp to avoid overwriting
timestamp = str(time.time()).split(".")[-1]
output_filename = f"{output_folder}{output_filename}_{timestamp}.csv"
print(f"Data will be saved in the file: {output_filename}" )


# Create the folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Folder created: {output_folder}")
else:
    print(f"Folder already exists: {output_folder}")
```

```
Data will be saved in the file: AD_dataset/data_training_7372458.csv
Folder already exists: AD_dataset/
```

Now, deploy the Oscilloscope instrument on your Moku. Connect to your Moku hardware by initializing a MultiInstrument with the device's IP address. In this example we use a Moku:Pro, and so we set platform_id to 4.

Connections in Multi-Instrument Mode are defined by an array of dictionaries. In this example, we route the signal from the analog frontend (Input 1) to the first input channel of the Oscilloscope. We also define the coupling, attenuation, and impedance of the frontend.

Note: Replace '10.1.119.245' with your Moku’s IP address (and adjust platform_id if using different Moku hardware).

```python
# Connect to the Moku device and deploy an Oscilloscope instrument
mim = MultiInstrument('10.1.119.245', force_connect=True, platform_id=4)
osc = mim.set_instrument(4, Oscilloscope)

connections = [dict(source="Input1", destination="Slot4InA")]
mim.set_connections(connections=connections)
mim.set_frontend(channel=1, impedance="50Ohm", coupling="DC", attenuation="-20dB")
```

```
{'attenuation': '-20dB', 'coupling': 'DC', 'impedance': '50Ohm'}
```
Configure the Oscilloscope acquisition parameters, including number of samples and how many frames of data to capture

```python
# Acquisition parameters
window_size = 100        # number of samples per frame (window length)
n_frames = 100           # number of frames to capture for dataset
t1 = 0.0                 # start time (seconds)
t2 = 0.1                 # end time (seconds) for each frame
duration = t2 - t1

# Set the oscilloscope timebase for each frame capture
osc.set_timebase(t1, t2)
```

```
{'t1': 0.0, 't2': 0.1}
```

Now collect the data frames from the Oscilloscope via the get_data command. We will capture n_frames frames of channel data and save them to the CSV file defined above. The code below reads data from the Oscilloscope, uses our save_data_to_csv function to append each frame to the file, and prints progress as it goes.

```python
for i in tqdm(range(n_frames), desc="Saving data"):
    data = osc.get_data()
    save_data_to_csv(data["ch1"], window_size, filename=output_filename)
```

```
Saving data:   0%|          | 0/100 [00:00<?, ?it/s]
```
Now let's verify the sampling rate and examine one captured frame of data as an example

```python
# Compute and display the sampling rate based on captured data
sampling_rate = osc.get_samplerate()
print("Sampling rate:", format_sampling_rate(osc.get_samplerate()["sample_rate"]))

# Plot an example frame of the captured data
plt.figure(figsize=(12, 4))
plt.plot(data["time"], data["ch1"])
plt.title("Example Captured Waveform (one frame)")
plt.xlabel("Time (s)")
plt.ylabel("Voltage")
plt.grid(True)
plt.show()
```
```
Sampling rate: 142.79 kHz
```

![png](./Anomaly_detection_files/Example_captured_waveform.png)

Print the first 100 datapoints of the frame. This 100-datapolint signal is an example of the training data that will be used for training the autoencoder.

```python
# Plot an example frame of the training data
plt.figure(figsize=(12, 4))
plt.plot(data["time"][:100], data["ch1"][:100])
plt.title("Example Training Data (one frame)")
plt.xlabel("Time (s)")
plt.ylabel("Voltage")
plt.grid(True)
plt.show()
```
![png](./Anomaly_detection_files/Example_training_data.png)

# Compose the training dataset

Now that data has been collected, we load it from the CSV and split it into separate training and testing sets. We will use a simple 80/20 split.

```python
# Read the saved frames from CSV
data = load_data_from_csv(output_filename)

# Split into training and testing datasets (80% train, 20% test)
split_index = int(len(data) * 0.8)
full_training_dataset = data[:split_index]
full_testing_dataset = data[split_index:]

print(f"N. of frames in the training dataset: {len(full_training_dataset)}")
print(f"N. of frames in the testing dataset: {len(full_testing_dataset)}")
```

```
N. of frames in the training dataset: 880
N. of frames in the testing dataset: 220
```

# Define the model and train

Instantiate the Moku Neural Network model object using LinnModel.

```python
quant_mod = LinnModel()
```

Set up an early stopping point for training to avoid overfitting. This will truncate the training if the validation loss does not improve for 10 epochs.

```python

early_stopping_config = {
    'patience': 10,            # allow 10 epochs without improvement
    'restore_best_weights': True,
    'monitor':"val_loss"
}
```

Now, prepare the model definition. This is an autoencoder, a small fully-connected neural network that compresses and then reconstructs the input signal. The model will output the same number of samples as the input frame (100). We disable automatic scaling of data (scale=False) because we want to work with raw values.

```python
frame_length = len(full_training_dataset[0])
print(f"frame length/input dimension: {frame_length}")

# Configure the model inputs/outputs (autoencoder: outputs = inputs)
quant_mod.set_training_data(training_inputs=full_training_dataset,
                             training_outputs=full_training_dataset,
                             scale=False)

# Define a simple autoencoder architecture: 
#  - encoder layers with 64, 32, 16 neurons (tanh activations)
#  - final decoder layer with 'frame_length' neurons (linear activation)
model_definition = [
    (64, 'tanh'),
    (32, 'tanh'),
    (16, 'tanh'),
    (frame_length, 'linear')
]

# Build the model
quant_mod.construct_model(model_definition)
```

```
frame length/input dimension: 100
WARNING:tensorflow:From c:\Users\miche\AppData\Local\Programs\Python\Python311\Lib\site-packages\keras\src\backend\tensorflow\core.py:222: The name tf.placeholder is deprecated. Please use tf.compat.v1.placeholder instead.
```
Now train the model on the training dataset we defined earlier. We include a 10% validation split (which comes from the training set itself) for early stopping.

```python
history = quant_mod.fit_model(
    epochs=250,
    es_config=early_stopping_config,
    validation_split=0.1,
    verbose=1
)
```

```
Value for restore missing. Using default:False.
Epoch 1/250
25/25 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.0013 - val_loss: 7.8818e-04
Epoch 2/250
25/25 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 6.5409e-04 - val_loss: 2.8883e-04
Epoch 3/250
25/25 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 2.1046e-04 - val_loss: 6.5728e-05
Epoch 4/250
25/25 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 5.0799e-05 - val_loss: 2.1085e-05
Epoch 5/250
25/25 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 1.4292e-05 - val_loss: 1.6994e-05
Epoch 6/250
25/25 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 1.7835e-05 - val_loss: 1.6665e-05
Epoch 7/250
25/25 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 1.4787e-05 - val_loss: 1.6296e-05
Epoch 8/250
25/25 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 1.7975e-05 - val_loss: 1.5801e-05
Epoch 9/250
25/25 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 1.0365e-05 - val_loss: 1.5625e-05
Epoch 10/250
25/25 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 9.7039e-06 - val_loss: 1.5155e-05
Epoch 11/250
25/25 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 9.5350e-06 - val_loss: 1.4581e-05
Epoch 12/250
25/25 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 9.9137e-06 - val_loss: 1.4012e-05
Epoch 13/250
25/25 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 1.0148e-05 - val_loss: 1.3402e-05
Epoch 14/250
25/25 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 1.1134e-05 - val_loss: 1.2680e-05
Epoch 15/250
25/25 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 9.4043e-06 - val_loss: 1.1916e-05
Epoch 16/250
25/25 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 1.2953e-05 - val_loss: 1.1168e-05
Epoch 17/250
25/25 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 1.2701e-05 - val_loss: 1.0311e-05
Epoch 18/250
25/25 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 9.0256e-06 - val_loss: 9.5096e-06
Epoch 19/250
25/25 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 1.0628e-05 - val_loss: 8.6078e-06
Epoch 20/250
25/25 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 7.2170e-06 - val_loss: 7.6696e-06
Epoch 21/250
25/25 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 1.0654e-05 - val_loss: 6.8180e-06
Epoch 22/250
25/25 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 7.2483e-06 - val_loss: 5.9443e-06
Epoch 23/250
25/25 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 6.8481e-06 - val_loss: 5.1698e-06
Epoch 24/250
25/25 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 5.5210e-06 - val_loss: 4.4748e-06
Epoch 25/250
25/25 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 5.2451e-06 - val_loss: 3.8111e-06
Epoch 26/250
25/25 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 3.8459e-06 - val_loss: 3.4560e-06
Epoch 27/250
25/25 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 4.1066e-06 - val_loss: 2.7537e-06
Epoch 28/250
25/25 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 4.6303e-06 - val_loss: 2.3208e-06
Epoch 29/250
25/25 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 4.3269e-06 - val_loss: 1.9804e-06
Epoch 30/250
25/25 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 3.3572e-06 - val_loss: 1.6821e-06
Epoch 31/250
25/25 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 3.7038e-06 - val_loss: 1.4242e-06
Epoch 32/250
25/25 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 2.0513e-06 - val_loss: 1.2224e-06
Epoch 33/250
25/25 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 1.8574e-06 - val_loss: 1.0894e-06
Epoch 34/250
25/25 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 2.0399e-06 - val_loss: 9.1873e-07
Epoch 35/250
25/25 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 1.7246e-06 - val_loss: 8.3453e-07
Epoch 36/250
25/25 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 2.0258e-06 - val_loss: 7.9282e-07
Epoch 37/250
25/25 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 1.0468e-06 - val_loss: 7.2050e-07
Epoch 38/250
25/25 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 1.0121e-06 - val_loss: 6.2446e-07
Epoch 39/250
25/25 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 1.3272e-06 - val_loss: 5.5632e-07
Epoch 40/250
25/25 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 8.7374e-07 - val_loss: 5.2139e-07
Epoch 41/250
25/25 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 1.5157e-06 - val_loss: 5.0296e-07
Epoch 42/250
25/25 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 7.7813e-07 - val_loss: 5.0497e-07
Epoch 43/250
25/25 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 4.8862e-07 - val_loss: 5.2613e-07
Epoch 44/250
25/25 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 8.6710e-07 - val_loss: 4.6760e-07
Epoch 45/250
25/25 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 8.6277e-07 - val_loss: 4.4028e-07
Epoch 46/250
25/25 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 6.0885e-07 - val_loss: 4.4799e-07
Epoch 47/250
25/25 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 6.5383e-07 - val_loss: 4.8272e-07
Epoch 48/250
25/25 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 8.2871e-07 - val_loss: 4.1005e-07
Epoch 49/250
25/25 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 5.2703e-07 - val_loss: 4.0595e-07
Epoch 50/250
25/25 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 4.0638e-07 - val_loss: 5.4495e-07
Epoch 51/250
25/25 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 5.0017e-07 - val_loss: 5.2292e-07
Epoch 52/250
25/25 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 5.3705e-07 - val_loss: 5.4037e-07
Epoch 53/250
25/25 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 5.2362e-07 - val_loss: 3.9301e-07
Epoch 54/250
25/25 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 4.7575e-07 - val_loss: 4.2096e-07
Epoch 55/250
25/25 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 3.7337e-07 - val_loss: 4.2197e-07
Epoch 56/250
25/25 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 3.9371e-07 - val_loss: 4.4283e-07
Epoch 57/250
25/25 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 4.0926e-07 - val_loss: 3.9660e-07
Epoch 58/250
25/25 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 2.8312e-07 - val_loss: 4.5293e-07
Epoch 59/250
25/25 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 3.5371e-07 - val_loss: 4.3901e-07
Epoch 60/250
25/25 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 4.5390e-07 - val_loss: 4.3006e-07
Epoch 61/250
25/25 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 3.8480e-07 - val_loss: 5.1594e-07
Epoch 62/250
25/25 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 3.4140e-07 - val_loss: 4.6303e-07
Epoch 63/250
25/25 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 5.4634e-07 - val_loss: 6.4741e-07
```

Plot the training and validation loss history to confirm convergence.

```python
plt.semilogy(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train loss', 'val loss'])
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.show()
```

![png](./Anomaly_detection_files/Validation_loss.png)

In this case, the final loss values are extremely low, indicating the autoencoder learned to reconstruct our sine wave signals very accurately. Next, save the quantized model to a file using save_linn. This will produce a .linn file that can be deployed to the Moku device’s FPGA for real-time inference.

```python
model_filename = "AD_model.linn"
# Create a time-base array for mapping (same length as frame)
T = np.linspace(-1, 1, frame_length)
# Save the trained model to file
save_linn(quant_mod, input_channels=1, output_channels=1,
          file_name=model_filename, output_mapping=[T.size-1])
```

```
Skipping layer 0 with type <class 'keras.src.layers.core.input_layer.InputLayer'>
Skipping layer 2 with type <class 'moku.nn._linn.OutputClipLayer'>
Skipping layer 4 with type <class 'moku.nn._linn.OutputClipLayer'>
Skipping layer 6 with type <class 'moku.nn._linn.OutputClipLayer'>
Skipping layer 8 with type <class 'moku.nn._linn.OutputClipLayer'>
Network latency approx. 224 cycles
```

# Testing the model - reconstruct the training dataset

Now that the model is trained, let’s verify how well it performs on data it saw during training. We feed the entire training dataset through the model to get reconstructed signals.

```python
full_training_dataset_np = np.array(full_training_dataset)
reconstructions = quant_mod.predict(full_training_dataset_np, scale=False, unscale_output=False)
```

```
28/28 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step 
```

The reconstructions array contains the model output for each input frame. For a well-trained autoencoder, these reconstructed frames should closely match the originals. As a quick visual check, we can pick one training frame and compare it with its reconstruction

```python
frame_id = 0  # for example, take the first training frame
plt.figure(figsize=(12, 4))
plt.plot(full_training_dataset[frame_id], label='Original')
plt.plot(reconstructions[frame_id], label='Reconstruction', linestyle='--')
plt.legend()
plt.title('Original vs Reconstructed Signal')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.grid(True)
plt.tight_layout()
plt.show()
```

![png](./Anomaly_detection_files/Original_vs_recon.png)

# Testing the model on the unseen dataset

Now we evaluate the autoencoder on the frames the model has not seen before. We expect the model to reconstruct normal frames well, but frames with anomalies (such as glitches) should yield a larger reconstruction error since the model was not trained on that behavior.

First, generate reconstructions for all test frames.

```python
full_testing_dataset_np = np.array(full_testing_dataset)
reconstructions_test = quant_mod.predict(full_testing_dataset_np)
```

```
7/7 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step 
```

To illustrate the model’s performance, let's compare the output on a specific test frame that contains a known anomaly versus a test frame that is normal. Suppose we know that frame index 147 in our dataset contains a glitch, and frame 146 is a normal frame for comparison.

```python
frame_id_anomaly_test = 31
frame_id_normal_test = 146 

# Plot original vs reconstructed for the anomalous frame
plt.figure(figsize=(12, 4))
plt.plot(full_testing_dataset[frame_id_anomaly_test], label='Original')
plt.plot(reconstructions_test[frame_id_anomaly_test], label='Reconstruction', linestyle='--')
plt.legend()
plt.title('Original vs Reconstructed Signal (Anomalous frame)')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot original vs reconstructed for a normal frame
plt.figure(figsize=(12, 4))
plt.plot(full_testing_dataset[frame_id_normal_test], label='Original')
plt.plot(reconstructions_test[frame_id_normal_test], label='Reconstruction', linestyle='--')
plt.legend()
plt.title('Original vs Reconstructed Signal (Normal frame)')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.grid(True)
plt.tight_layout()
plt.show()
```

![png](./Anomaly_detection_files/Anomalous_frame.png)
![png](./Anomaly_detection_files/Normal_frame.png)

To detect the anomaly quantitatively, we need to examine the reconstruction error.

# Extract the reconstruction error

Calculate the element-wise error between original and reconstructed signals for the test set.

```python
original = np.array(full_testing_dataset)
reconstructed = np.array(reconstructions_test)

# --- Error calculations ---
# Absolute error for each sample
absolute_error = np.abs(original - reconstructed)
# Squared error for each sample
squared_error = (original - reconstructed) ** 2

# Summed error per frame (could be used as an anomaly score per frame)
absolute_error_per_frame = absolute_error.sum(axis=1)
squared_error_per_frame = squared_error.sum(axis=1)
```
Now, compare the error profiles of the previously chosen anomalous frame with the normal frame. We expect the frame with the glitch to show a spike in error at the glitch location, whereas the normal frame’s error should remain near zero across all sample indices. For example, we can plot the squared error for both frames on the same graph.

```python
plt.figure(figsize=(12, 4))
plt.plot(squared_error[frame_id_anomaly_test], label='Frame with anomaly', color='orange')
plt.plot(squared_error[frame_id_normal_test], label='Normal Frame', color='blue')
plt.title('Reconstruction Squared Error (per sample)')
plt.xlabel('Index')
plt.ylabel('Error')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

![png](./Anomaly_detection_files/Reconstruction_error.png)

In the plot above, the anomalous frame (orange curve) has several spikes in error, whereas the normal frame error (blue line) is essentially flat at zero. This indicates the autoencoder was unable to reconstruct those glitch points, as expected.

We can further quantify this by applying a numerical threshold. In this example we'll set this threshold at an absolute error of 0.0002.

```python
threshold = 0.0002

# Find indices in the anomaly frame where absolute error > threshold
indices_anom = [i for i, v in enumerate(squared_error[frame_id_anomaly_test]) if v > threshold]
values_anom = [squared_error[frame_id_anomaly_test][i] for i in indices_anom]
print("Frame with anomaly")
print("Indices of values > threshold:", indices_anom)
print("Corresponding values:", values_anom)

print("—" * 70)

# Find indices in the normal frame where absolute error > threshold
indices_norm = [i for i, v in enumerate(squared_error[frame_id_normal_test]) if v > threshold]
values_norm = [squared_error[frame_id_normal_test][i] for i in indices_norm]
print("Normal frame")
print("Indices of values > threshold:", indices_norm)
print("Corresponding values:", values_norm)
```

```
Frame with anomaly
Indices of values > threshold: []
Corresponding values: []
——————————————————————————————————————————————————————————————————————
Normal frame
Indices of values > threshold: []
Corresponding values: []
```

As expected, the anomalous frame has a few samples where the reconstruction error exceeds the threshold, while the normal frame has none. These correspond to the glitch points in the signal.

To further reduce false detections, we experimented with using higher-order error metrics. For example, defining a "focal MSE loss" (squared error raised to a power >1) can down-weight small differences and emphasize larger errors.

```python
def focal_mse_loss(prediction, target, gamma=2.5):
    error = prediction - target
    squared_error = error ** 2
    return squared_error ** gamma

# Compute focal loss for each sample
focal_loss = focal_mse_loss(reconstructed, original)
focal_loss_per_frame = focal_loss.sum(axis=1)
```

```python
plt.figure(figsize=(12, 4))
plt.plot(focal_loss[frame_id_anomaly_test], label='Frame with anomaly', color='orange')
plt.plot(focal_loss[frame_id_normal_test], label='Normal Frame', color='blue')
plt.title('Reconstruction Focal Loss (per sample)')
plt.xlabel('Index')
plt.ylabel('Error')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

![png](./Anomaly_detection_files/Recon_focal_loss.png)

```python
threshold = 1e-9

# Find indices where value > threshold

# Frame with Anomaly
indices = [i for i, v in enumerate(focal_loss[frame_id_anomaly_test]) if v > threshold]

# Print results
print("Frame with anomaly")
print("Indices of values > threshold:", indices)
print("Corresponding values:", [focal_loss[frame_id_anomaly_test][i] for i in indices])


print("-------------------------------------------------------------------------------")


# Normal with Anomaly
indices = [i for i, v in enumerate(focal_loss[frame_id_normal_test]) if v > threshold]

# Print results
print("Normal frame")
print("Indices of values > threshold:", indices)
print("Corresponding values:", [focal_loss[frame_id_normal_test][i] for i in indices])
```

```
Frame with anomaly
Indices of values > threshold: []
Corresponding values: []
-------------------------------------------------------------------------------
Normal frame
Indices of values > threshold: []
Corresponding values: []
```

Overall, the autoencoder approach successfully learned the normal signal behavior and flagged the anomaly via reconstruction error. In summary:

* Normal frames: Reconstructed almost perfectly, with near-zero error across all samples.
* Anomalous frame: Reconstruction deviates at the glitch points, yielding measurable error spikes.

This demonstrates how an FPGA-deployable autoencoder, trained via Moku’s Python API, can perform real-time anomaly detection on signal data. The .linn model file saved can be loaded onto the Moku device to monitor incoming signals and detect anomalies based on the model’s reconstruction error in real time.

# Deploy on Moku

To test the model and deploy the anomaly detection feature, open your Moku on Multi-Instrument Mode.

Place the Neural Network instrument into Slot 1 and load the .linn file. Route the signal source to the Neural Network input (e.g., the output of a Waveform Generator or an external signal source), and enable the output. The Neural Network will output the reconstructed signal in real time.

You can then place an Oscilloscope in another slot to visualize both the original and reconstructed signals simultaneously. You should see that the reconstructed signal closely tracks the original except when anomalies occur, as we observed in the offline test. The reconstructed signal might be slightly delayed relative to the original due to some finite latency. One strategy to compensate for this is to deploy a second Neural Network instrument configured as an identity network (a network that simply outputs its input) in another slot to delay the original signal by the same amount. By aligning the phase in this way, you can directly compare the original and autoencoder output signals.

To automate anomaly detection, you can measure the difference between the original and reconstructed signals on the Moku. For example, you could use a custom Moku Cloud Compile (MCC) module that outputs the squared error. This error signal will spike when the original signal deviates from the learned normal pattern, which can be detected using a threshold trigger on the Oscilloscope or Data Logger.

![png](./Anomaly_detection_files/AD_MiM.png)

![png](./Anomaly_detection_files/AD_error.png)
