#
# moku example: Basic Datalogger streaming
#
# This example demonstrates use of the Datalogger instrument to
# stream time-series voltage data and plot it using matplotlib
#
# (c) Liquid Instruments Pty. Ltd.
#
import matplotlib.pyplot as plt

from moku.instruments import Datalogger

# force_connect will overtake an existing connection
i = Datalogger('192.168.###.###', force_connect=True)

try:
    # generate a waveform on output channel 1
    i.generate_waveform(1, "Sine", frequency=100)

    # disable Input2 as we want to stream data only from Input1
    i.enable_input(2, enable=False)

    # set the sample rate to 10KSa/s
    # stream the data for 10 seconds..
    i.start_streaming(duration=10, sample_rate=10e3)

    # Set up the plotting parameters
    plt.ion()
    plt.show()
    plt.grid(visible=True)
    plt.ylim([-1, 1])

    (line1,) = plt.plot([])

    # Configure labels for axes
    ax = plt.gca()

    # This loops continuously updates the plot with new data
    while True:
        # get the chunk of streamed data
        data = i.get_stream_data()
        if data:
            plt.xlim([data['time'][0], data['time'][-1]])
            # Update the plot
            line1.set_ydata(data['ch1'])
            line1.set_xdata(data['time'])
            plt.pause(0.001)

except Exception as e:
    if str(e) == "End of stream":
        print("Streaming session complete!")
    else:
        i.stop_streaming()
        i.relinquish_ownership()
        raise e
finally:
    i.relinquish_ownership()
