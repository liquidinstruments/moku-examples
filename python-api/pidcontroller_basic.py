# moku example: Basic PID Controller
#
# This script demonstrates how to configure one of the two PID Controllers
# in the PID Controller instrument. Configuration is done by specifying
# frequency response characteristics of the controller.
#
# (c) Liquid Instruments Pty. Ltd.
#

from moku.instruments import PIDController

# Connect to your Moku by its ip address using PIDController('192.168.###.###')
# force_connect will overtake an existing connection
i = PIDController('192.168.###.###', force_connect=True)

try:
    # Configures the control matrix:
    # Channel 1: input 1 gain = 1 dB, input 2 gain = 0 dB
    # Channel 2: input 2 gain = 0 dB, input 2 gain = 1 dB
    i.set_control_matrix(channel=1, input_gain1=1, input_gain2=0)
    i.set_control_matrix(channel=2, input_gain1=0, input_gain2=1)

    # Configure PID Control loop 1 using frequency response characteristics
    #   P = -10dB
    #   I Crossover = 100Hz
    #   D Crossover = 10kHz
    #   I Saturation = 10dB
    #   D Saturation = 10dB
    #   Double-I = OFF
    i.set_by_frequency(
        channel=1,
        prop_gain=-10,
        int_crossover=1e2,
        diff_crossover=1e4,
        int_saturation=10,
        diff_saturation=10,
    )

    # Configure PID Control loop 2 using gain
    #   Proportional gain = 10
    #   Differentiator gain = -5
    #   Differentiator gain corner = 5 kHz
    i.set_by_gain(channel=2, overall_gain=0, prop_gain=10, diff_gain=-5, diff_corner=5e3)

    # Enable the outputs of the PID controller
    i.enable_output(1, signal=True, output=True)
    i.enable_output(2, signal=True, output=True)

except Exception as e:
    i.relinquish_ownership()
    raise e
finally:
    # Close the connection to the Moku device
    # This ensures network resources and released correctly
    i.relinquish_ownership()
