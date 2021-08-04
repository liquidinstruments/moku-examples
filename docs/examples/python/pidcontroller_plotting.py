# moku example: PID Controller Plotting Example
#
# This script demonstrates how to configure both PID Controllers
# in the PID Controller instrument. Configuration on the Channel 1
# PID is done by specifying frequency response characteristics,
# while Channel 2 specifies the gain characteristics.
#
# The output response of each PID Controller channel is plotted
# in real-time.
#
# (c) 2021 Liquid Instruments Pty. Ltd.
#
import matplotlib.pyplot as plt
from moku.instruments import PIDController

# Connect to your Moku by its ip address using PIDController('192.168.###.###')
# or by its serial number using PIDController(serial=123)
i = PIDController('192.168.###.###', force_connect=False)

try:
    # Configures the control matrix:
    # Channel 1: input 1 gain = 1 dB, input 2 gain = 0 dB
    # Channel 2: input 2 gain = 0 dB, input 2 gain = 1 dB
    # Matrix mapping will be updated in the next release
    i.set_control_matrix(1,1,1)
    i.set_control_matrix(2,0,0)

    # Configure the Channel 1 PID Controller using frequency response
    # characteristics
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

    # Configure the Channel 2 PID Controller using gain characteristics
    #   Overall Gain = 6dB
    #   I Gain       = 20dB 
    i.set_by_gain(channel=2, overall_gain=6.0, prop_gain=20)

    # Set the probes to monitor Output 1 and Output 2
    i.set_monitor(1, 'Output1')
    i.set_monitor(2, 'Output2')

    # Set the timebase
    i.set_timebase(-1e-3, 1e-3) # +- 1msec
    i.set_trigger(type='Edge', source='Input1', level=0)

    # Enable the output channels of the PID controller
    i.enable_output(1, True, True)
    i.enable_output(2, True, True)

    # Get initial data frame to set up plotting parameters. This can be done
    # once if we know that the axes aren't going to change (otherwise we'd do
    # this in the loop)
    data = i.get_data()

    # Set up the plotting parameters
    plt.ion()
    plt.show()
    plt.grid(b=True)
    plt.ylim([-1, 1])
    plt.xlim([data['time'][0], data['time'][-1]])

    line1, = plt.plot([])
    line2, = plt.plot([])

    # Configure labels for axes
    ax = plt.gca()

    # This loops continuously updates the plot with new data
    while True:
        # Get new data
        data = i.get_data()

        # Update the plot
        line1.set_ydata(data['ch1'])
        line2.set_ydata(data['ch2'])
        line1.set_xdata(data['time'])
        line2.set_xdata(data['time'])
        plt.pause(0.001)

except Exception as e:
    print(f'Exception occurred: {e}')
finally:
    # Close the connection to the Moku device
    # This ensures network resources and released correctly
    i.relinquish_ownership()
