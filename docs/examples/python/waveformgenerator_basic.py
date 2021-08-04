#
# moku example: Waveform Generator Basic
#
# This example demonstrates how you can use the Waveform Generator
# instrument to generate a sinewave on Channel 1 and a squarewave on Channel 2.
#
# (c) 2021 Liquid Instruments Pty. Ltd.
#
from moku.instruments import WaveformGenerator

# Connect to your Moku by its ip WaveformGenerator('192.168.###.###')
# or by its serial m = WaveformGenerator(serial=123)
i = WaveformGenerator('192.168.###.###', force_connect=False)

try:
    # Generate a sine wave on channel 1, 0.5 Vpp, 5 kHz
    # Generate a square wave on channel 2, 1 Vpp, 1 kHz, 50% duty cycle
    i.generate_waveform(channel=1, type='Sine', amplitude=0.5, frequency=5e3)
    i.generate_waveform(channel=2, type='Square', amplitude=1.0, frequency=1e3, duty=50)

except Exception as e:
    print(f'Exception occurred: {e}')
finally:
    # Close the connection to the Moku device
    # This ensures network resources and released correctly
    i.relinquish_ownership()
