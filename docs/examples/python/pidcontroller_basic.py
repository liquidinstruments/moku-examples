# moku example: Basic PID Controller
#
# This script demonstrates how to configure one of the two PID Controllers
# in the PID Controller instrument. Configuration is done by specifying
# frequency response characteristics of the controller.
#
# (c) 2021 Liquid Instruments Pty. Ltd.
#

from moku.instruments import PIDController

# Connect to your Moku by its ip address using PIDController('192.168.###.###')
# or by its serial number using PIDController(serial=123)
i = PIDController('192.168.###.###', force_connect=False)

try:
    # Configure the PID Controller using frequency response characteristics
    # 	P = -10dB
    #	I Crossover = 100Hz
    # 	D Crossover = 10kHz
    # 	I Saturation = 10dB
    # 	D Saturation = 10dB
    # 	Double-I = OFF
    # Note that gains must be converted from dB first
    i.set_by_frequency(channel=1, prop_gain=-10, int_crossover=1e2,
                       diff_crossover=1e4, int_saturation=10,
                       diff_saturation=10)
    
    i.enable_output(1, signal=True, output=True)

except Exception as e:
    print(f'Exception occurred: {e}')
finally:
    # Close the connection to the Moku device
    # This ensures network resources and released correctly
    i.relinquish_ownership()
