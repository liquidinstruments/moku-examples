#
# Python moku example: Plotting Digital Filter Box
#
# This example demonstrates how you can configure the Digital
# Filter Box instrument to filter and display two signals. Filter
# 1 takes its input from Input1 and applies a lowpass Butterworth
# filter. Filter 2 takes its input from Input1 + Input2 and applies
# a highpass Elliptic filter. The output of both filters are then
# displayed on the plot.
#
# (c) Liquid Instruments Pty. Ltd.
#

import matplotlib.pyplot as plt

from moku import MokuException
from moku.instruments import DigitalFilterBox

# Connect to your Moku by its ip address using
# DigitalFilterBox('192.168.###.###')
# force_connect will overtake an existing connection
i = DigitalFilterBox('192.168.2.125', force_connect=True)

try:
    # Set Channel 1 and 2 to DC coupled, 1 MOhm impedance, and
    # default input range (400 mVpp range on Moku:Pro, 1 Vpp
    # range on Moku:Lab, 10 Vpp range on Moku:Go)
    i.set_frontend(1, coupling='DC', impedance='1MOhm',
                   attenuation='0dB')
    i.set_frontend(2, coupling='DC', impedance='1MOhm',
                   attenuation='0dB')

    # Channel1 signal: Input 1
    # Channel2 signal: Input 1 + Input 2
    i.set_control_matrix(1, input_gain1=1, input_gain2=0)
    i.set_control_matrix(2, input_gain1=1, input_gain2=1)

    # Configure Digital Filter on Channel 1 and Channel 2
    # Channel1 is a 8th-order Butterworth lowpass filter
    # Channel2 is a 8th-order Elliptic highpass filter
    # 3.906 MHz is for Moku:Go
    # Please change sampling rate for other Moku devices.
    i.set_filter(1, "3.906MHz", shape="Lowpass",
                 type="Butterworth", low_corner=1e3,
                 order=8)
    i.set_filter(2, "3.906MHz", shape="Highpass",
                 type="Elliptic", high_corner=100e3,
                 order=8)

    # Monitor ProbeA: Filter Channel1 output
    # Monitor ProbeB: Filter Channel2 output
    i.set_monitor(1, "Output1")
    i.set_monitor(2, "Output2")

    # Enable Digital Filter Box output ports
    i.enable_output(1, signal=True, output=True)
    i.enable_output(2, signal=True, output=True)

    # View +- 0.5 ms i.e. trigger in the centre
    i.set_timebase(-0.5e-3, 0.5e-3)

    # Set up the plotting parameters
    plt.ion()
    plt.show()
    plt.grid(visible=True)
    plt.ylim([-0.01, 0.01])

    line1, = plt.plot([], label="Lowpass Filter Output")
    line2, = plt.plot([], label="Highpass Filter Output")

    # Configure labels for axes
    ax = plt.gca()
    ax.legend(handles=[line1, line2], loc=1)
    plt.xlabel("Time [Second]")
    plt.ylabel("Amplitude [Volt]")

    # This loops continuously updates the plot with new data
    while True:
        # Get new data
        data = i.get_data()
        if data:
            plt.xlim([data['time'][0], data['time'][-1]])
            # Update the plot
            line1.set_ydata(data['ch1'])
            line2.set_ydata(data['ch2'])
            line1.set_xdata(data['time'])
            line2.set_xdata(data['time'])
            plt.pause(0.001)
except MokuException as e:
    print("Couldn't configure Moku. Please check your IP address and that you've updated the script parameters (such as sampling rate) to match your device.")
    raise e
finally:
    # Releasing ownership of the Moku allows other users to connect
    # to it without forcing a takeover.
    i.relinquish_ownership()
