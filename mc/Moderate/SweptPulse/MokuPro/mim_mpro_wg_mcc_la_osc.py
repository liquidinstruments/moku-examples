# Example code to 
# 1. Configure Moku:Pro in multi-instrument mode
# 2. Import custom MCC design that will
#	a. Create a custom frequency and duty cycle pulse
#	b. Pass through an input to an output when the pulse is high
# 3. Use the custom pulse to trigger swept waveform modulation
# 4. Use Datalogger to record pulse samples for further analysis and signal processing
# 5. Setup Oscilloscope to display results

# To execute without change, a single Moku:Pro is required.  This example will internally route the signals within Multi-instrument Mode  
#	Configuration - 
#   - Route Bus1 to Slot1 InputA
#	- Route Slot 1 OutputA to Slot 2 InputB
#   - Route Slot 2 OutputB to Slot 1 InputA (this will get implicitly mapped to an available Bus)
#	- Route Slot 2 OutputA to Slot3 & 4 InputA
#   - Route Slot 2 OutputB to Slot3 & 4 InputB

# Date last edited - 6 Jan 2025
#
# (c) 2025 Liquid Instruments Pty. Ltd.


# Import the needed libraries 
from moku.instruments import MultiInstrument
from moku.instruments import CloudCompile, WaveformGenerator, Oscilloscope, Datalogger

import time

# Establish connection to Moku:Pro
mp = MultiInstrument('192.168.X.X', force_connect=True, platform_id=4) #Edit IP for your device

try:

	bitstream = "./bitstreams.tar" #edit for the filename of your bitstream if different
	wg = mp.set_instrument(1, WaveformGenerator)
	mcc = mp.set_instrument(2, CloudCompile, bitstream=bitstream)
	dl = mp.set_instrument(3, Datalogger)
	osc = mp.set_instrument(4, Oscilloscope)


	# # Configure Moku:Pro with MCC in MiM
	connections = [dict(source="Slot1OutA", destination="Slot2InB"),
					dict(source="Slot2OutB", destination="Slot1InA"),
					dict(source="Slot2OutA", destination="Slot3InA"),
					dict(source="Slot2OutB", destination="Slot3InB"),
					dict(source="Slot4InA", destination="Slot2OutA"),
					dict(source="Slot4InB", destination="Slot2OutB")
					]
	mp.set_connections(connections=connections)

	sfreq = input("Enter the swept pulse starting frequency (in Hz): ")
	efreq = input("Enter the swept pulse ending frequency (in Hz): ")
	PRF = input("Enter the desired Pulse Repetition Frequency (in Hz): ")
	duty = input("Enter the desired Pulse duty cycle (in percentage from 0 to 100): ")
	sweepT = (1/float(PRF))*(float(duty)/100) #Calculate the sweep time for the pulse
	wg.generate_waveform(channel=1, type='Sine', amplitude=2, frequency=float(sfreq))
	wg.set_sweep_mode(channel=1, source='InputA', trigger_level=.1, stop_frequency=float(efreq), sweep_time=float(sweepT) )


	# Adjusted for the clock frequency of the Moku:Pro
	freqControl = int(312500000/float(PRF))
	dutyControl = int(freqControl*float(duty)/100)

	mcc.set_control(0,freqControl)
	mcc.set_control(1,dutyControl)


	# Set the sample rate to 500 KSa/s
	dl.set_samplerate(500e3)

	# Set the acquisition mode
	dl.set_acquisition_mode(mode='Normal')


	# stream data for 200ms (this should get 2 full pulses based on the 100 Hz PRI)
	# trigger is only used to keep all of the data files aligned in time.  
	dl.start_streaming(.2,trigger_source='InputB',trigger_level=.01)
	dl.stream_to_file('dataStream.csv')	
	time.sleep(5)
	dl.stop_streaming()


finally:
	# Close the connection to the Moku device
	# This ensures network resources and released correctly
	mp.relinquish_ownership()



