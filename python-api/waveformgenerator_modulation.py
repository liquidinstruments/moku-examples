#
# moku example: Waveform Generator Modulation
#
# This example demonstrates how you can use the Waveform Generator instrument
# to generate an amplitude modulated sine wave on Channel 1, and a sweep
# modulated sine wave on Channel 2.
#
# (c) Liquid Instruments Pty. Ltd.
#
from moku.instruments import WaveformGenerator

# Connect to your Moku by its ip address using WaveformGenerator('192.168.###.###')
# force_connect will overtake an existing connection
i = WaveformGenerator('192.168.###.###', force_connect=True)

try:
    # Generate a sine wave on channel 1, 0.5 Vpp, 5 kHz
    # Generate a sine wave on channel 2, 1 Vpp, 1 MHz
    i.generate_waveform(channel=1, type='Sine', amplitude=0.5, frequency=5e3)
    i.generate_waveform(channel=2, type='Sine', amplitude=1.0, frequency=1e6)

    # Configure amplitude modulation on channel 1.
    # Use internal reference as modulation source, modulation depth 50%,
    # modulated at a frequency of 1Hz
    i.set_modulation(channel=1, type='Amplitude', source='Internal', depth=50, frequency=1)

    # Configure Channel 2 with sweep trigger modulation.
    # Use Input 1 as trigger source, trigger level is 0.1 V.
    # Start the sweep at waveform frequency 1 MHz and  stop at 10 Hz, each sweep is 3 seconds.
    i.set_sweep_mode(
        channel=2, source='Input1', stop_frequency=10.0, sweep_time=3.0, trigger_level=0.1
    )

except Exception as e:
    i.relinquish_ownership()
    raise e
finally:
    # Close the connection to the Moku device
    # This ensures network resources and released correctly
    i.relinquish_ownership()
