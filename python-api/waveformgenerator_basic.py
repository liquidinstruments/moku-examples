#
# moku example: Waveform Generator Basic
#
# This example demonstrates how you can use the Waveform Generator
# instrument to generate a sinewave on Channel 1 and a squarewave on Channel 2.
#
# (c) Liquid Instruments Pty. Ltd.
#
from moku.instruments import WaveformGenerator

# Replace with your Moku's IP address.
DEVICE_IP = '192.168.###.###'

# Connect to the Moku (force_connect will overtake an existing connection).
with WaveformGenerator(DEVICE_IP, force_connect=True) as i:
    # Generate a sine wave on channel 1, 0.5 Vpp, 5 kHz
    # Generate a square wave on channel 2, 1 Vpp, 1 kHz, 50% duty cycle
    i.generate_waveform(channel=1, type='Sine', amplitude=0.5, frequency=5e3)
    i.generate_waveform(channel=2, type='Square', amplitude=1.0, frequency=1e3, duty=50)
