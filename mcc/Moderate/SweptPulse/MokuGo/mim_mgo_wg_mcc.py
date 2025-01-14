# Example code to 
# 1. Configure single Moku:Go in multi-instrument mode
# 2. Import custom MCC design that will
#	a. Create a custom frequency and duty cycle pulse
#	b. Pass through an input to an output when the pulse is high
# 3. Use the custom pulse to trigger swept waveform modulation

# To execute without change, this example will require access to a single Moku:Go  
#	In order to display the results, you will need another oscilloscope or data logger

# Date last edited - 13 Jan 2025
#
# (c) 2025 Liquid Instruments Pty. Ltd.


# Import the needed libraries 
from moku.instruments import MultiInstrument
from moku.instruments import CloudCompile, WaveformGenerator

# Establish connection to Moku:Go - Waveform Generator and MCC
mg1 = MultiInstrument('192.168.1.136', force_connect=True, platform_id=2)

try:
	# Configure Moku:Go to generate pulsed signal with variable frequency and duty cycle

	bitstream = "./pcMask601.tar"
	wg = mg1.set_instrument(1, WaveformGenerator)
	mcc = mg1.set_instrument(2, CloudCompile, bitstream=bitstream)

	# Configure Moku:Go with waveform generator and MCC in MiM
	connections = [dict(source="DIO", destination="Slot2InA"),
					dict(source="Slot2OutB", destination="Slot1InA"),
					dict(source="Slot2OutA", destination="DIO"),
					dict(source="Slot2OutB", destination="Output1"),
					dict(source="Slot2OutC", destination="Output2"),
					dict(source="Slot1OutA", destination="Slot2InB")
					]
	mg1.set_connections(connections=connections)
	mg1.set_dio(direction=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])

	sfreq = input("Enter the swept pulse starting frequency (in Hz): ")
	efreq = input("Enter the swept pulse ending frequency (in Hz): ")
	PRF = input("Enter the desired Pulse Repetition Frequency (in Hz): ")
	duty = input("Enter the desired Pulse duty cycle (in percentage from 0 to 100): ")
	sweepT = (1/float(PRF))*(float(duty)/100) #Calculate the sweep time for the pulse to ensure full sweep per pulse
	wg.generate_waveform(channel=1, type='Sine', amplitude=2, frequency=float(sfreq))
	wg.set_sweep_mode(channel=1, source='InputA', trigger_level=.1, stop_frequency=float(efreq), sweep_time=float(sweepT) )

	freqControl = int(31250000/float(PRF))
	dutyControl = int(freqControl*float(duty)/100)

	mcc.set_control(0,freqControl)
	mcc.set_control(1,dutyControl)


finally:
	# Close the connection to the Moku devices
	# This ensures network resources and released correctly
	mg1.relinquish_ownership()



