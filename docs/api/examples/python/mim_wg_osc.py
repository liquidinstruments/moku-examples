import matplotlib.pyplot as plt

from moku.instruments import MultiInstrument
from moku.instruments import Oscilloscope, WaveformGenerator

m = MultiInstrument('192.168.###.###', platform_id=2)
try:
    wg = m.set_instrument(1, WaveformGenerator)
    osc = m.set_instrument(2, Oscilloscope)

    connections = [dict(source="Input1", destination="Slot1InA"),
                   dict(source="Slot1OutA", destination="Slot2InA"),
                   dict(source="Slot1OutA", destination="Slot2InB"),
                   dict(source="Slot2OutA", destination="Output1")]

    print(m.set_connections(connections=connections))

    wg.generate_waveform(1, "Sine")
    osc.set_timebase(-5e-3, 5e-3)
    data = osc.get_data()

    # Set up the plotting parameters
    plt.ion()
    plt.show()
    plt.grid(visible=True)
    plt.ylim([-1, 1])
    plt.xlim([data['time'][0], data['time'][-1]])

    line1, = plt.plot([])
    line2, = plt.plot([])

    # Configure labels for axes
    ax = plt.gca()

    # This loops continuously updates the plot with new data
    while True:
        # Get new data
        data = osc.get_data()

        # Update the plot
        line1.set_ydata(data['ch1'])
        line2.set_ydata(data['ch2'])
        line1.set_xdata(data['time'])
        line2.set_xdata(data['time'])
        plt.pause(0.001)

except Exception as e:
    raise e
finally:
    # Close the connection to the Moku device
    # This ensures network resources and released correctly
    m.relinquish_ownership()
