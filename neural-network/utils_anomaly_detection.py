import ast
import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random

def generate_clean_sine_100(x1=1, x2=100, n_points=100, amplitude=1, phase_shift=1, random_parameters=True, print_values=False):
    if random_parameters:
        amplitude = random.randint(0, 120)
        phase_shift = random.randint(5, 45)
        if print_values:
            print('phase_shift', phase_shift)
        
    t_out = np.linspace(x1, x2, n_points)
    v_out = amplitude * np.sin((t_out - phase_shift * np.pi / 4) / (2 * np.pi))
    
    if print_values:
        print('phase shift:', phase_shift, "amplitude:", amplitude)
    
    return v_out, t_out, amplitude


def add_gaussian_noise(signal, snr_db=None, snr_db_range=(25, 30)):
    """
    Adds Gaussian noise to a signal at a random or specified SNR in dB.

    Parameters:
    - signal (numpy array): Input signal.
    - snr_db (float or None): Desired SNR in dB. If None, random value from snr_db_range is used.
    - snr_db_range (tuple): Min and max range for random SNR (in dB).

    Returns:
    - noisy_signal (numpy array): Signal with added Gaussian noise.
    - snr_db (float): The SNR in dB used to generate the noise.
    """
    if snr_db is None:
        snr_db = random.uniform(*snr_db_range)

    signal_power = np.mean(signal ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)

    noisy_signal = signal + noise
    return noisy_signal, snr_db


def add_glitch_gaussian(
    v_in, 
    t_out, 
    amplitude, 
    glitch_factor=0.5, 
    glitch_width=30, 
    index_pos=-1, 
    mean=0.5, 
    random_variables=False, 
    save_to_file=False, 
    csv_file="dataset_tests/glitch_data.csv"
):
    """
    Adds a sharp Gaussian dip glitch to the signal.

    Returns:
        v_out_glitch: signal with glitch
        relative_width: glitch width / signal length
        glitch_factor: depth of the glitch applied
    """
    v_out_glitch = v_in.copy()
    glitch_center = len(v_in) // 2  # Always insert in the middle for consistency

    if random_variables:
        std_dev = 0.1
        glitch_factor = max(0.1, min(0.9, random.gauss(mean, std_dev)))  # Clamp to avoid flat signal

    # Create a Gaussian dip (negative peak)
    x = np.linspace(-1, 1, glitch_width)
    gaussian_dip = 1 - glitch_factor * np.exp(-((x) ** 2) / (2 * 0.1 ** 2))  # Sharp, centered dip

    # Insert into signal
    start = glitch_center - glitch_width // 2
    end = start + glitch_width
    v_out_glitch[start:end] *= gaussian_dip

    if save_to_file:
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([index_pos, start, glitch_width, glitch_factor])

    return v_out_glitch, glitch_width / len(v_in), glitch_factor

def generate_noisy_sine_wave(amplitude_value, phase_shift_value, snr_percentage_value):
    
    phase_shift_value += random.uniform(0,phase_shift_value*0.05)
    v_out, t_out, amplitude = generate_clean_sine_100(amplitude=amplitude_value, phase_shift=phase_shift_value, random_parameters=False)
    signal, snr_percentage = add_gaussian_noise(v_out)

    return signal

def generate_noisy_sine_wave_with_glitch(amplitude_value, phase_shift_value, index_position, csv_file, glitch_factor_value = 2, mean=0.125):
    phase_shift_value += random.uniform(0,phase_shift_value*0.05)
    
    v_out, t_out, amplitude = generate_clean_sine_100(amplitude=amplitude_value, phase_shift=phase_shift_value, random_parameters=False)
    signal,  glitch_width_p, glitch_factor = add_glitch_gaussian(v_out, t_out, amplitude, glitch_factor=glitch_factor_value, random_variables=False, index_pos=index_position, mean=mean, save_to_file=True, csv_file=csv_file)
    signal, snr_percentage = add_gaussian_noise(signal)

    return signal

def create_dataset_glitch(num_samples=5000, amplitude_value=1, anomaly_percentage=0.01 , glitch_factor_value=0.8, folder_path="training_dataset_ad/"):

    data_file_name = "data.csv"
    metadata_file_name = "anomaly_metadata.csv"

    # Randomise the index for the anomalous frames
    n_anomaly_samples = int(num_samples * anomaly_percentage)
    print(f"{n_anomaly_samples}/{num_samples} frames will peresent an anomaly")

    # Select unique random anomaly indexes
    anomaly_indexes = random.sample(range(num_samples), n_anomaly_samples)

    # Create the folder and the output files
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")


    anomaly_file_value = folder_path + metadata_file_name

    # Define headers
    headers_anomaly = ["sample_n", "glitch_start", "width", "factor"]
        
    # Create the anomaly file and write the headers
    with open(anomaly_file_value, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers_anomaly)

    headers_metadata = ["sample_n", "type", "data"]    
    metadata_file_value = folder_path + data_file_name

    # Create the metadat file and write the headers
    with open(metadata_file_value, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers_metadata)


    signals = []
    snr_percentage_value = random.uniform(0.1, 0.6) # This

    for i in range(num_samples):
            
        phase_shift_value = random.randint(5, 45) # The phase shift is randomized for each frame 
                
        if i in anomaly_indexes: # This is a frame with an anomaly
            signal =generate_noisy_sine_wave_with_glitch(amplitude_value, phase_shift_value, index_position=i, csv_file=anomaly_file_value, glitch_factor_value=glitch_factor_value, mean=glitch_factor_value)
            signal_type = 'anomaly'
        else:
            signal = generate_noisy_sine_wave(amplitude_value, phase_shift_value, snr_percentage_value)
            signal_type = "normal"

        signals.append(signal)
        # Flatten signal to a list and write to CSV
        with open(metadata_file_value, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([i + 1, signal_type, list(map(float, signal))])

    return signals, anomaly_indexes


def focal_mse_loss_numpy(prediction, target, gamma=2.0):
    error = prediction - target
    squared_error = error ** 2
    focal_weight = (np.abs(error)) ** gamma
    loss =  focal_weight * squared_error
    return loss


def focal_mse_loss(prediction, target, gamma=2.5):
    error = prediction - target
    squared_error = error ** 2
    loss = squared_error ** gamma
    return loss

def calculate_reconstruction_error_by_class_focal(original, reconstructed, frame_anomaly):
    focal_anomaly = []
    focal_normal = []
    focal_total = []

    for i in range(len(reconstructed)):
        loss = focal_mse_loss(reconstructed[i], original[i])
        loss_mean = loss.mean()
        if i in frame_anomaly:
            focal_anomaly.append(loss_mean)
        else:
            focal_normal.append(loss_mean)
        focal_total.append(loss_mean)

    return focal_anomaly, focal_normal, focal_total

def load_dataset(folder_path):

    # Print the list of all files in the dataser folder - it should contain 2 CSV files
    all_files = os.listdir(folder_path)
    print(f"Files in the folder {folder_path} : {all_files}")
    prefix_data = "data"  
    prefix_anomaly = "anomaly"

    waveform_metadata_file = next(file for file in all_files if file.startswith(prefix_data))
    print(f"Waveform metadata filename: {waveform_metadata_file}")

    anomaly_metadata_file = next(file for file in all_files if file.startswith(prefix_anomaly))
    print(f"Anomaly metadata filename: {anomaly_metadata_file}")

    waveform_matadata_path = f"{folder_path}{waveform_metadata_file}"
    anomaly_matadata_path = f"{folder_path}{anomaly_metadata_file}"

    dataset = SineWaveDataset(file_path=waveform_matadata_path)

    return dataset, waveform_matadata_path, anomaly_matadata_path

def plot_reconstruction(original, reconstructed, frame_n):
    
    original_signal = original
    reconstructed_signal = reconstructed

    print(frame_n, original_signal[frame_n][:5])
    
    mse = np.mean((original_signal[frame_n] - reconstructed_signal[frame_n]) ** 2)
    max_v = np.max((original_signal[frame_n] - reconstructed_signal[frame_n]) ** 2)
    mse_values = (original_signal[frame_n] - reconstructed_signal[frame_n]) ** 2
    print("Mean Squared Error:", mse)
    print("Max MSE:", max_v)
    
    # Plot the original and reconstructed signals on the same plot
    plt.figure(figsize=(14, 6))

    # Original signal
    plt.plot(original_signal[frame_n], label='Original Signal')
    
    # Reconstructed signal
    plt.plot(reconstructed_signal[frame_n], label='Reconstructed Signal', color='orange', linestyle='--')

    # Add labels and title
    plt.title('Original vs Reconstructed Signal')
    plt.xlabel('Time Steps')
    plt.ylabel('Signal Amplitude')
    plt.legend()

    # Display the plot
    plt.show()


    def run_inference(model, testing_data):
        reconstructed_data = model(testing_data)
        
        return reconstructed_data


def run_inference(model, testing_data):
    reconstructed_data = model(testing_data)
    
    return reconstructed_data


def check_anomaly_from_metadata(csv_path, element_list, column_name='sample_n'):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Check if the specified column exists
    if column_name not in df.columns:
        return False, f"Column '{column_name}' not found in the CSV file."
    
    # Check if any element from the list is in the column
    matches = []
    for element in element_list:
        if element in df[column_name].values:
            matches.append(element)
    
    # Return results
    if matches:
        return True, f"Found {len(matches)} anomalies"
    else:
        return False, f"No anomalies found"


class SineWaveDataset:
    
    def __init__(self, num_samples=5000, training_dataset_with_anomaly=False, file_path="sine_wave_dataset.csv"):
        self.data = []
        self._load_data_from_csv(file_path)

    def _load_data_from_csv(self, file_path):
        """
        Load sine wave signals and labels from a CSV file.
        The CSV contains: Index, Type, and the 1000 signal points.
        """
        with open(file_path, mode="r") as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header row
            for row in reader:
                try:
                    # Clean the signal string if necessary, removing unwanted spaces or characters
                    row_list = ast.literal_eval(row[2])
                    self.data.append(row_list)
                except Exception as e:
                    print(f"Error processing row: {row}. Error: {e}")
                    break
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    