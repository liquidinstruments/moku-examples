#
# moku example: Plotting Time and Frequency Analyzer
#
# This example demonstrates how you can configure the Time and Frequency Analyzer
# instrument, and view histogram data frames in real-time.
#
# (c) Liquid Instruments Pty. Ltd.
#
import matplotlib.pyplot as plt
import numpy as np

from moku.instruments import TimeFrequencyAnalyzer

# Connect to your Moku by its ip address using TimeFrequencyAnalyzer('192.168.###.###')
# force_connect will overtake an existing connection
i = TimeFrequencyAnalyzer('192.168.###.###', force_connect=True)

try:
    # Set the event detectors
    # Set Event A to detect rising edge events on Input 1 at 0V
    # Set Event B to detect rising edge events on Input 2 at 0V
    i.set_event_detector(1, 'Input1', threshold=0, edge='Rising')
    i.set_event_detector(2, 'Input2', threshold=0, edge='Rising')

    # Set the interpolation to Linear
    i.set_interpolation(mode='Linear')

    # Set acquisition mode to a 100ms Window
    i.set_acquisition_mode(mode='Windowed', window_length=100e-3)

    # Set the histogram to 8ns span centred around 2us
    i.set_histogram(start_time=1.996e-6, stop_time=2.004e-6)

    # Set the interval analyzers
    # Set Interval A to start at Event A and stop at Event A
    # Set Interval B to start at Event B and stop at Event B
    i.set_interval_analyzer(1, start_event_id=1, stop_event_id=1)
    i.set_interval_analyzer(2, start_event_id=2, stop_event_id=2)

    # Get initial data frame to set up plotting parameters. This can be done
    # once if we know that the axes aren't going to change (otherwise we'd do
    # this in the loop)
    data = i.get_data()

    # Set up the plotting parameters
    t0 = data['interval1']['histogram']['t0']
    dt = data['interval1']['histogram']['dt']
    length = 1024
    x = np.linspace(start=t0, stop=t0 + dt * 1023, num=1024)

    plt.ion()
    plt.show()
    plt.grid(visible=True)
    plt.ylim([0, 600])
    plt.xlim([1.996e-6, 2.004e-6])

    line1 = plt.bar(
        x, height=data['interval1']['histogram']['data'], width=dt, align='edge', alpha=0.8
    )
    line2 = plt.bar(
        x, height=data['interval2']['histogram']['data'], width=dt, align='edge', alpha=0.8
    )

    # Configure labels for axes
    ax = plt.gca()

    # This loops continuously updates the plot with new data
    while True:
        # Get new data
        data = i.get_data()

        # Update the plot
        height1 = data['interval1']['histogram']['data']
        height2 = data['interval2']['histogram']['data']
        for count1, rect1, count2, rect2 in zip(height1, line1.patches, height2, line2.patches):
            rect1.set_height(count1)
            rect2.set_height(count2)
        plt.pause(0.001)

except Exception as e:
    i.relinquish_ownership()
    raise e
finally:
    # Close the connection to the Moku device
    # This ensures network resources and released correctly
    i.relinquish_ownership()
