#
# moku example: Basic Phasemeter
#
# This example demonstrates how you can configure the Phasemeter
# instrument to measure 4 independent signals.

# (c) 2022 Liquid Instruments Pty. Ltd.
#

from moku.instruments import Phasemeter

# Connect to your Moku by its ip address using Phasemeter('192.168.###.###')
# or by its serial number using Phasemeter(serial=123)
i = Phasemeter('192.168.###.###', force_connect=False)

try:
	# SetChannel 1 and 2 to DC coupled, 1 MOhm impedance, and 400 mVpp range
	i.set_frontend(1, coupling='DC', impedance='1MOhm', range='400mVpp')
	i.set_frontend(2, coupling='DC', impedance='1MOhm', range='400mVpp')
	i.set_frontend(3, coupling='DC', impedance='1MOhm', range='400mVpp')
	i.set_frontend(4, coupling='DC', impedance='1MOhm', range='400mVpp')

	# Configure Output channel 1 to generate sine waves at 1 Vpp, 2 MHz
	i.generate_output(1, 'Sine', amplitude=1, frequency=2e6)
	# Configure Output channel 2 to be phase locked to Input 2 signal at an
	# amplitude of 0.5 Vpp
	i.generate_output(2, 'Sine', amplitude=0.5, phase_locked=True)
	# Configure Output channel 3 and 4 to generate measured phase at a
	# scaling of 1 V/cycle and 10 V/cycle respectively
	i.generate_output(3, 'Phase', scaling=1)
	i.generate_output(4, 'Phase', scaling=10)

	# Set the acquisition speed to 596Hz for all channels
	i.set_acquisition_speed('596Hz')

	# Set all input channels to 2 MHz, bandwidth 100 Hz
	i.set_pm_loop(1, auto_acquire=False, frequency=2e6, bandwidth='100Hz')
	i.set_pm_loop(2, auto_acquire=False, frequency=2e6, bandwidth='100Hz')
	i.set_pm_loop(3, auto_acquire=False, frequency=2e6, bandwidth='100Hz')
	i.set_pm_loop(4, auto_acquire=False, frequency=2e6, bandwidth='100Hz')

	# Get all the data available from the Moku
	data = i.get_data();


except Exception as e:
	print(f'Exception occurred: {e}')
finally:
	# Close the connection to the Moku device
	# This ensures network resources are released correctly
	i.relinquish_ownership()