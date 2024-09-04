#
# moku example: Waveform Generator Triggering
#
# This example demonstrates how you can use the Waveform Generator instrument
# to generate a gated sine wave on Channel 1, and a swept frequency square wave
# on Channel 2.
#
# (c) 2024 Liquid Instruments Pty. Ltd.
#

from moku.instruments import WaveformGenerator

# Connect to your Moku by its ip address using WaveformGenerator('192.168.###.###')
# or by its serial number using WaveformGenerator(serial=123)
i = WaveformGenerator('192.168.###.###', force_connect=False)

try:
    # Set sine wave to channel 1 and square wave to channel 2
    i.generate_waveform(channel=1, type='Sine',
                        amplitude=1, frequency=10)
    i.generate_waveform(channel=2, type='Square',
                        amplitude=1, frequency=500, duty=50)

    # Activate burst trigger for output 1 and sweep trigger for output 2
    i.set_burst_mode(channel=1, source='Internal', mode='Gated',
                     burst_period=2, trigger_level=0.5, burst_duration=0.1)
    i.set_sweep_mode(channel=2, source='Internal', stop_frequency=10,
                     sweep_time=2, trigger_level=0.1)

except Exception as e:
    print(f'Exception occurred: {e}')
finally:
    # Close the connection to the Moku device
    # This ensures network resources are released correctly
    i.relinquish_ownership()
