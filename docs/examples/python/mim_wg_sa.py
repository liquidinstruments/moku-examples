import matplotlib.pyplot as plt

from moku.instruments import MultiInstrument
from moku.instruments import SpectrumAnalyzer, WaveformGenerator

m = MultiInstrument("192.168.###.###", platform_id=2)
try:
    w = m.set_instrument(1, WaveformGenerator)
    s = m.set_instrument(2, SpectrumAnalyzer)

    connections = [dict(source="Input1", destination="Slot1InA"),
                   dict(source="Slot1OutA", destination="Slot2InA")]

    print(m.set_connections(connections=connections))

    w.generate_waveform(1, "Sine", frequency=10e5)
    s.set_span(frequency1=0, frequency2=10e5)
    s.set_rbw('Auto')

    line1, = plt.plot([])
    plt.ion()
    plt.show()
    plt.grid(visible=True)
    plt.autoscale(axis='x', tight=True)

    # Get an initial frame of data to set any frame-specific plot parameters
    frame = s.get_data()

    # Format the x-axis as a frequency scale
    ax = plt.gca()

    # Get and update the plot with new data
    while True:
        frame = s.get_data()

        # Set the frame data for each channel plot
        line1.set_ydata(frame['ch1'])
        # Frequency axis shouldn't change, but to be sure
        line1.set_xdata(frame['frequency'])
        # Ensure the frequency axis is a tight fit
        ax.relim()
        ax.autoscale_view()

        # Redraw the lines
        plt.draw()
        plt.pause(0.001)

except Exception as e:
    raise e
finally:
    # Close the connection to the Moku device
    # This ensures network resources and released correctly
    m.relinquish_ownership()
