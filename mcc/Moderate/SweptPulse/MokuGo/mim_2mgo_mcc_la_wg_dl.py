# Example code to 
# 1. Configure two Moku:Go's in multi-instrument mode
# 2. Import custom MCC design that will
#	a. Create a custom frequency and duty cycle pulse
#	b. Pass through an input to an output when the pulse is high
# 3. Use the custom pulse to trigger swept waveform modulation

# To execute without change, this example will require access to two Moku:Go's.  
#	Configuration - 
#	- Connect Out1 of Moku:Go #1 to In1 of Moku:Go #2
#	- Connect Out2 of Moku:Go #1 to In2 of Moku:Go #2
#	- Connect Out1 of Moku:Go #2 to In1 of Moku:Go #1

# Date last edited - 12 Jan 2025
#
# (c) 2025 Liquid Instruments Pty. Ltd.


# Import the needed libraries 
from moku.instruments import MultiInstrument
from moku.instruments import CloudCompile, LogicAnalyzer, WaveformGenerator, Datalogger

# # Import libraries for matplotlib in order to plot results
import matplotlib.pyplot as plt

import time

# import time

# # Establish connection to Moku:Go #1 - MCC and Logic Analyzer
mg1 = MultiInstrument('192.168.1.136', force_connect=True, platform_id=2)

# Establish connection to Moku:Go
# mg2 = MultiInstrument('192.168.1.37', force_connect=True, platform_id=2)

try:
	# Configure Moku:Go to generate pulsed signal with variable frequency and duty cycle

	bitstream = "./pcMask601.tar"
	wg = mg1.set_instrument(1, WaveformGenerator)
	mcc = mg1.set_instrument(2, CloudCompile, bitstream=bitstream)
	# osc = mg2.set_instrument(1, LogicAnalyzer)
	# dl = mg2.set_instrument(2, Datalogger)

	# Configure Moku:Go #1 with MCC and logic analyzer in MiM
	connections = [dict(source="DIO", destination="Slot2InA"),
					dict(source="Slot2OutB", destination="Slot1InA"),
					dict(source="Slot2OutA", destination="DIO"),
					dict(source="Slot2OutB", destination="Output1"),
					dict(source="Slot2OutC", destination="Output2"),
					dict(source="Slot1OutA", destination="Slot2InB")
					]
	mg1.set_connections(connections=connections)
	mg1.set_dio(direction=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])

	# # Configure Moku:Go #2 with waveform generator and oscilloscope in MiM
	# connectionsMP = [dict(source="Input2", destination="Slot1InA"),
	# 				dict(source="Slot1OutA", destination="Output1"),
	# 				# dict(source="Slot1OutA", destination="Slot2InA"),
	# 				dict(source="Input1", destination="Slot2InA"),
	# 				dict(source="Input2", destination="Slot2InB")
	# 				]
	# mg2.set_connections(connections=connectionsMP)

	sfreq = input("Enter the swept pulse starting frequency (in Hz): ")
	efreq = input("Enter the swept pulse ending frequency (in Hz): ")
	PRF = input("Enter the desired Pulse Repetition Frequency (in Hz): ")
	duty = input("Enter the desired Pulse duty cycle (in percentage from 0 to 100): ")
	sweepT = (1/float(PRF))*(float(duty)/100) #Calculate the sweep time for the pulse
	wg.generate_waveform(channel=1, type='Sine', amplitude=2, frequency=float(sfreq))
	wg.set_sweep_mode(channel=1, source='InputA', trigger_level=.1, stop_frequency=float(efreq), sweep_time=float(sweepT) )
	# print(wg.summary())

	# Set the sample rate to 500 KSa/s
	# dl.set_samplerate(250e3)

	# Set the acquisition mode
	# dl.set_acquisition_mode(mode='Normal')


	# stream data for 200ms (this should get 2 full pulses based on the 100 Hz PRI)
	# trigger is only used to keep all of the data files aligned in time.  Occasionally noise will result  
	# in a failure to record data.  By adjusting cnt above you can reproduce a single data file.
	# dl.start_streaming(.2,trigger_source='InputB',trigger_level=.01)
	# dl.stream_to_file('dataStream.csv')
	# time.sleep(5)
	# dl.stop_streaming()


	freqControl = int(31250000/float(PRF))
	# print(freqControl)
	dutyControl = int(freqControl*float(duty)/100)
	# print(dutyControl)

	mcc.set_control(0,freqControl)
	mcc.set_control(1,dutyControl)


finally:
	# Close the connection to the Moku devices
	# This ensures network resources and released correctly
	mg1.relinquish_ownership()
	# mg2.relinquish_ownership()


