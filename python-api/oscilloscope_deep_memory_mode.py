## Example: Data Acquisition with Deep Memory Mode in the Moku Oscilloscope
#
# This example demonstrates how to acquire data using the deep memory mode in the Moku Oscilloscope.
#
# The 'save_high_res_buffer' command stores high-resolution channel buffer data in Moku's internal storage.
#
# Ensure that deep memory mode is enabled using the 'set_acquisition_mode' command before exporting high-res data.
#
# Logged data files can be retrieved from the following storage locations:
#   - 'persist' for Moku:Go
#   - 'tmp' for Moku:Lab
#   - 'ssd' for Moku:Pro
#   - 'ssd' for Moku:Delta
# Update the 'download' command accordingly to match the correct storage location.
#
# The parameters in the 'set_frontend' command should be configured to align with the specific hardware (Moku:Go, Moku:Lab, Moku:Pro, or Moku:Delta).


import matplotlib.pyplot as plt
import os
import time
import numpy as np

from moku.instruments import Oscilloscope

# Connect to your Moku by its ip address using Oscilloscope('192.168.###.###')
# force_connect will overtake an existing connection
i = Oscilloscope('192.168.###.###', force_connect=True)

NUM_FRAMES = 1  # This is the number of frames to be averaged
FILE_PATH = "C:/Users/XXXX/Downloads" # Please replace with your own FILE_PATH

try:
    # Set the data source of Channel 1 to be Input 1   
    i.set_frontend(1,'50Ohm','DC','400mVpp')
    i.set_sources([{"channel": 1, "source": "Input1"},
                   {"channel": 2, "source": "None"},
                   {"channel": 3, "source": "None"},
                   {"channel": 4, "source": "None"}])

    i.set_trigger(mode='Normal', type='Edge', source='Input1', level=0)
    i.set_timebase(-5e-3, 20e-3)
    i.set_acquisition_mode('DeepMemory')
    print(i.get_samplerate())

    for iter in range(NUM_FRAMES):
        i.get_data(wait_reacquire=True, wait_complete=True)
        response = i.save_high_res_buffer(comments="Triggered")
        file_name = response["file_name"]
        temp_filename = FILE_PATH + "/high_res_data-" + time.strftime('%d-%m-%Y-%H_%M_%S')
        i.download("ssd", file_name, temp_filename + ".li")
        os.system("mokucli convert --format=npy " + temp_filename + ".li")
        file = np.load(temp_filename + ".npy")
        data_ch1 = file['Channel A (V)']
        if iter == 0:
            ch1 = data_ch1
        else:
            ch1 = ch1 + data_ch1
            
        # (Optional) Delete the downloaded and converted files
        os.remove(temp_filename + ".npy")
        os.remove(temp_filename + ".li")
        os.remove(temp_filename + ".txt")
    
    time_column = file['Time (s)']

    # plot the average of all acquired high-res frames
    plt.plot(time_column, ch1/NUM_FRAMES)
    plt.grid(visible=True)
    
    # Configure labels and ranges for axes
    ax = plt.gca()
    ax.set_xlim([time_column[0], time_column[-1]])
    ax.set_ylim([-1, 1])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage (V)")
    plt.show()

except Exception as e:
    i.relinquish_ownership()
    raise e
finally:
    # Close the connection to the Moku device
    # This ensures network resources and released correctly
    i.relinquish_ownership()
