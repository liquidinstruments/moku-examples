# To execute without change, this example will require access to both a Moku:Pro and a Moku:Go.  
#	Configuration - Connect Out1 of Moku:Go to In1 of Moku:Pro
#
# Example code to 
# 1. Generate example of pulsed signal (i.e. pulsed RADAR) using waveform generator on Moku:Go - Output to channel 1
# 2. Use Multi Instrument Mode on Moku:Pro to 
#	a. Create additive noise (Slot 1 - Waveform Generator)
#	b. Scale and combine noise to signal received from Moku:Go (Slot 2 - PID Controller)
#		NOTE: PID Controller is only used for the control matrix to combine noise with pulsed signal
#	c. Use Autoencoder Neural Network to de-noise the noisey pulsed signal (Slot 3 - Neural Network)
# 	d. Log Data (non-noisey pulse, noisey pulse, de-noised pulse) using Data Logger (Slot 4 - Data Logger)
# 3. Execute a loop to gradually increase the signal level on Moku:Go to vary Signal to Noise Ratio 
# 4. Data is stored in separate .csv file for each signal level for post processing 

# Date last edited - 2 Dec 2024
#
# (c) 2024 Liquid Instruments Pty. Ltd.


# Import the needed libraries 
from moku.instruments import MultiInstrument
from moku.instruments import WaveformGenerator, PIDController, NeuralNetwork, Datalogger

# Import libraries for matplotlib in order to plot results
import matplotlib.pyplot as plt

import time

# Establish connection to Moku:Pro
m = MultiInstrument('192.168.1.226', force_connect=True, platform_id=4)

# Establish connection to Moku:Go
mg = WaveformGenerator('192.168.1.47', force_connect=True)

try:
	# Configure Moku:Go to generate pulsed signal
	# Pulse parameters are:
		# Pulse Repetition Interval (PRI) - 10 Hz
		# Carrier Frequency - 2kHz
		# Will start with an output of 1Vpp and gradually decrease
	mg.generate_waveform(channel=1, type='Sine', amplitude=.1, frequency=2e3)
	mg.generate_waveform(channel=2, type='Pulse', pulse_width=.005, edge_time=16e-9, amplitude=1, frequency=10)
	mg.set_modulation(channel=1, type='Amplitude', source='Output2', depth=200)

	# Configure Moku:Pro with Neural Network in MiM
	wg = m.set_instrument(1, WaveformGenerator)
	pid = m.set_instrument(2, PIDController)
	nn = m.set_instrument(3, NeuralNetwork)
	dl = m.set_instrument(4, Datalogger)
	connections = [dict(source="Slot1OutB", destination="Slot2InB"),
					dict(source="Input1", destination="Slot2InA"),
					dict(source="Slot2OutA", destination="Slot3InA"),
					dict(source="Slot2OutA", destination="Slot3InB"),
					dict(source="Slot2OutA", destination="Slot3InC"),
					dict(source="Slot2OutA", destination="Slot3InD"),
					dict(source="Slot3OutA", destination="Slot4InD"),
					dict(source="Input1", destination="Slot4InB"),
					dict(source="Slot2OutA", destination="Slot4InC")
					]
	m.set_connections(connections=connections)

	# Configure Waveform Generator
	wg.generate_waveform(channel=2, type='Noise')

	# Configure PID Controller to combine noise with received pulsed waveform

	# # Configures the control matrix:
	# Channel 1: input 1 gain = 1 dB, input 2 gain = 20 dB
	pid.set_control_matrix(channel=1, input_gain1=1, input_gain2=20)

	# Configure PID Control loop 1 using frequency response characteristics
		# P = 0dB
	pid.set_by_frequency(channel=1, prop_gain=0)
	pid.enable_output(1, signal=True, output=True)
	print(pid.summary())

	# Load network into Neural Network and set inputs and outputs
	nn.set_input(strict=False, channel=1, low_level=-1, high_level=1)
	nn.set_input_sample_rate(sample_rate=305000)
	nn.upload_network("./autoencoder_32.linn")
	nn.set_output(strict=False, channel=1, enabled=True, low_level=-1, high_level=1)
	print(nn.summary())

	# Only log data from inputs 2,3,4 per the configuration above
	dl.enable_input(1, enable=False)

	# Set the sample rate to 500 KSa/s
	dl.set_samplerate(500e3)

	# Set the acquisition mode
	dl.set_acquisition_mode(mode='Normal')

	# Gradually increase input pulse from .1Vpp to a max of 2Vpp 
	for cnt in range(1,21,1):
		# stream data for 200ms (this should get 2 full pulses based on the 10 Hz PRI)
		# trigger is only used to keep all of the data files aligned in time.  Occasionally noise will result  
		# in a failure to record data.  By adjusting cnt above you can reproduce a single data file.
		dl.start_streaming(.2,trigger_source='InputB',trigger_level=.01)
		dl.stream_to_file('dataStream'+str(cnt)+'.csv')
		time.sleep(5)
		dl.stop_streaming()
		mg.generate_waveform(channel=1, type='Sine', amplitude=cnt*.1, frequency=2e3)

finally:
	# Close the connection to the Moku device
	# This ensures network resources and released correctly
	mg.relinquish_ownership()
	m.relinquish_ownership()
	# try:
	# 	m.relinquish_ownership()
	# except:
	# 	time.sleep(2)
	# 	m.relinquish_ownership()

