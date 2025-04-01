import matplotlib.pyplot as plt

from moku.instruments import Datalogger, LockInAmp, MultiInstrument

# force_connect will overtake an existing connection
i = MultiInstrument('192.168.###.###', platform_id=2, force_connect=True)

try:
    dl = i.set_instrument(1, Datalogger)
    lia = i.set_instrument(2, LockInAmp)

    connections = [dict(source="Input1", destination="Slot1InA"),
                   dict(source="Slot1OutA", destination="Slot2InA"),
                   dict(source="Slot1OutA", destination="Slot2InB"),
                   dict(source="Slot2OutA", destination="Output1")]

    i.set_connections(connections=connections)

    dl.generate_waveform(1, "Sine", frequency=1000)

    dl.start_streaming(10)

    dl.stream_to_file()

    lia.set_monitor(1, "Input1")

    lia.start_streaming(10)

    plt.ion()
    plt.show()
    plt.grid(visible=True)
    plt.ylim([-1, 1])

    line1, = plt.plot([])
    line2, = plt.plot([])

    # Configure labels for axes
    ax = plt.gca()

    # This loops continuously updates the plot with new data
    while True:
        # Get new data
        data = lia.get_stream_data()

        # Update the plot
        if data:
            plt.xlim([data['time'][0], data['time'][-1]])
            line1.set_ydata(data['ch1'])
            line1.set_xdata(data['time'])
            plt.pause(0.001)

except Exception as e:
    i.relinquish_ownership()
    raise e
finally:
    i.relinquish_ownership()
