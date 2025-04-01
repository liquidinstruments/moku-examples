#
# moku example: Basic Lock-in Amplifier
#
# This example demonstrates how you can configure the Lock-in Amplifier
#  instrument to demodulate an input signal from Input 1 with the reference
#  signal from the Local Oscillator to extract the X component and generate
#  a sine wave on the auxiliary output

# (c) Liquid Instruments Pty. Ltd.
#
from moku.instruments import LockInAmp

# Connect to your Moku by its ip address using LockInAmp('192.168.###.###')
# force_connect will overtake an existing connection
i = LockInAmp('192.168.###.###', force_connect=True)

try:
    # Set Channel 1 and 2 to DC coupled, 1 MOhm impedance, and 400 mVpp range
    i.set_frontend(1, coupling='DC', impedance='1MOhm', attenuation='0dB')
    i.set_frontend(2, coupling='DC', impedance='1MOhm', attenuation='0dB')

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

except Exception as e:
    i.relinquish_ownership()
    raise e
finally:
    # Close the connection to the Moku device
    # This ensures network resources are released correctly
    i.relinquish_ownership()
