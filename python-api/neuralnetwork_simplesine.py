#
# moku example: Neural Network
#
# This script demonstrates how to use the Neural Network instrument to generate
# a ramp wave and process it through the uploaded neural network, finally
# viewing the output in the oscilloscope. This uses the network generated
# in the Simple Sine wave example.
#
# (c) Liquid Instruments Pty. Ltd.

import matplotlib.pyplot as plt

from moku.instruments import MultiInstrument, NeuralNetwork, Oscilloscope, WaveformGenerator

# force_connect will overtake an existing connection
m = MultiInstrument('192.168.###.###', platform_id=4, force_connect=True)

try:
    # Set up MiM configuration
    wg = m.set_instrument(1, WaveformGenerator)
    nn = m.set_instrument(2, NeuralNetwork)
    osc = m.set_instrument(3, Oscilloscope)

    connections = [
        dict(source="Slot1OutA", destination="Slot2InA"),
        dict(source="Slot1OutA", destination="Slot2InB"),
        dict(source="Slot1OutB", destination="Slot2InC"),
        dict(source="Slot1OutB", destination="Slot2InD"),
        dict(source="Slot2OutA", destination="Slot3InA"),
    ]

    print(m.set_connections(connections=connections))

    # Generate ramp wave in Waveform Generator
    wg.generate_waveform(channel=1, type='Ramp', amplitude=0.25, frequency=5e3, symmetry=100)
    print(wg.summary())

    # Load network into Neural Network and set inputs and outputs
    nn.set_input(strict=False, channel=1, low_level=-1, high_level=1)
    nn.set_input_sample_rate(sample_rate=305000)
    nn.upload_network("/path/to/simple_sine_model.linn")
    nn.set_output(strict=False, channel=1, enabled=True, low_level=-1, high_level=1)
    print(nn.summary())

    # Set up Oscilloscope for plotting
    osc.set_timebase(-5e-3, 5e-3)
    data = osc.get_data()
    print(osc.summary())

    # Set up the plotting parameters
    plt.ion()
    plt.show()
    plt.grid(visible=True)
    plt.ylim([-1, 1])
    plt.xlim([data['time'][0], data['time'][-1]])

    (line,) = plt.plot([])

    # Configure labels for axes
    ax = plt.gca()

    # This loops continuously updates the plot with new data
    while True:
        # Get new data
        data = osc.get_data()

        # Update the plot
        line.set_ydata(data['ch1'])
        line.set_xdata(data['time'])
        plt.pause(0.001)

except Exception as e:
    m.relinquish_ownership()
    raise e
finally:
    # Close the connection to the Moku device
    # This ensures network resources and released correctly
    m.relinquish_ownership()
