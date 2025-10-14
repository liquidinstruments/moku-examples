#
# moku example: Plotting Lock-in Amplifier
#
# This example demonstrates how you can configure the Lock-in Amplifier
#  instrument to demodulate an input signal from Input 1 with the reference
#  signal from the Local Oscillator to extract the X component and generate
#  a sine wave on the auxiliary output

# (c) Liquid Instruments Pty. Ltd.
#

import matplotlib.pyplot as plt

from moku.instruments import LockInAmp

# Connect to your Moku by its ip address using LockInAmp('192.168.###.###')
# force_connect will overtake an existing connection
i = LockInAmp('192.168.###.###', force_connect=True)

try:
    # Set Channel 1 and 2 to DC coupled, 1 MOhm impedance, and 400 mVpp range
    i.set_frontend(1, coupling='DC', impedance='1MOhm', attenuation='0dB')
    i.set_frontend(2, coupling='DC', impedance='1MOhm', attenuation='-20dB')

    # Configure the demodulation signal to Local oscillator with 1 MHz and
    # 0 degrees phase shift
    i.set_demodulation('Internal', frequency=1e6, phase=0)

    # Set low pass filter to 1 kHz corner frequency with 6 dB/octave slope
    i.set_filter(1e3, slope='Slope6dB')

    # Configure output signals
    # X component to Output 1
    # Aux oscillator signal to Output 2 at 1 MHz 500 mVpp
    i.set_outputs('X', 'Aux')
    i.set_aux_output(1e6, 0.5)

    # Set up signal monitoring
    # Configure monitor points to Input 1 and main output
    i.set_monitor(1, 'Input1')
    i.set_monitor(2, 'MainOutput')

    # Configure the trigger conditions
    # Trigger on Probe A, rising edge, 0V
    i.set_trigger(type='Edge', source='ProbeA', level=0)

    # View +- 1 ms i.e. trigger in the centre
    i.set_timebase(-1e-3, 1e-3)

    # Get initial data frame to set up plotting parameters. This can be done
    # once if we know that the axes aren't going to change (otherwise we'd do
    # this in the loop)
    data = i.get_data()

    # Set up the plotting parameters
    plt.ion()
    plt.show()
    plt.grid(visible=True)
    plt.ylim([-1, 1])
    plt.xlim([data['time'][0], data['time'][-1]])

    (line1,) = plt.plot([])
    (line2,) = plt.plot([])

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
    i.relinquish_ownership()
    raise e
finally:
    # Close the connection to the Moku device
    # This ensures network resources are released correctly
    i.relinquish_ownership()
