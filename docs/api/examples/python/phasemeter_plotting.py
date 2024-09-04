#
# moku example: Plotting Phasemeter
#
# This example demonstrates how you can configure the Phasemeter instrument
# and collect live samples from it's output.
# The signal amplitude is calculated using these samples, and plotted for
# real-time viewing.
#
# (c) 2024 Liquid Instruments Pty. Ltd.
#

from moku.instruments import Phasemeter
import matplotlib.pyplot as plt
from time import time
import numpy as np

# Connect to your Moku by its ip address using Phasemeter('192.168.###.###')
# or by its serial number using Phasemeter(serial=123)
i = Phasemeter("192.168.###.###", force_connect=True)

try:
    # Set samplerate to 37 Hz/s
    i.set_acquisition_speed(speed="37Hz")

    # Set Channel 1 and 2 to DC coupled, 1 MOhm impedance, and 4Vpp range
    i.set_frontend(1, coupling="DC", impedance="1MOhm", range="4Vpp")
    i.set_frontend(2, coupling="DC", impedance="1MOhm", range="4Vpp")

    # Configure output channel 1 to generate sine waves at 0.5Vpp, 5 MHz
    i.generate_output(1, "Sine", amplitude=0.5, frequency=5e6)
    # Configure output channel 2 to generate sine waves at 1Vpp, 10 MHz
    i.generate_output(2, "Sine", amplitude=1, frequency=10e6)

    # Get auto acquired frequency for channel 1 and 2
    i.get_auto_acquired_frequency(channel=1)
    i.get_auto_acquired_frequency(channel=2)

    # Get initial data frame to set plot bounds
    data = i.get_data()

    # Set plot bounds
    plt.ion()
    plt.show()
    plt.grid(True)
    plt.ylabel("Amplitude (Vrms)")
    plt.xlabel("Time (s)")
    plt.xlim((-20, 0))

    (line1,) = plt.plot([])
    (line2,) = plt.plot([])
    timestamps = []
    ydata1 = []
    ydata2 = []

    # Configure labels for axes
    ax = plt.gca()

    # This loop continuously updates the plot with new data
    while True:
        # Get new data
        data = i.get_data()

        # Update the x- to mimic the Moku App traces
        # Append the current timestamp then normalize by the current time
        # so that now is t = 0s
        t_now = time()
        timestamps += [t_now]
        xdata = np.array(timestamps) - t_now

        # Update the plot
        ydata1 += [data["ch1"]["amplitude"]]
        ydata2 += [data["ch2"]["amplitude"]]
        line1.set_ydata(ydata1)
        line2.set_ydata(ydata2)
        line1.set_xdata(xdata)
        line2.set_xdata(xdata)

        ax.relim()
        ax.autoscale_view()
        plt.pause(0.001)
        plt.draw()

        # Ensure data is maintained to a reasonable length of 1000 points
        # This will cover more than the 20 second x-axis limit
        timestamps = timestamps[-1000:]
        ydata1 = ydata1[-1000:]
        ydata2 = ydata2[-1000:]

except Exception as e:
    print(f"Exception Occurred: {e}")
finally:
    # Close the connection to the Moku device
    # This ensures network resources are released correctly
    i.relinquish_ownership()