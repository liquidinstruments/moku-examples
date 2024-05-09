#
# moku example: Basic Laser Lock Box
#
# This example demonstrates how you can configure the Laser Lock Box
# Instrument and monitor the signals at Input 1 and Input 2.

# (c) 2022 Liquid Instruments Pty. Ltd.
#

from moku.instruments import LaserLockBox

# Connect to your Moku by its ip address using LaserLockBox('192.168.###.###')
# or by its serial number using LaserLockBox(serial=123)
i = LaserLockBox('192.168.xxx.xxx', force_connect=False)

try:
    # Set Channel 1 and 2 to DC coupled, 1 MOhm impedance, and 400 mVpp range
    i.set_frontend(1, coupling='DC', impedance='1MOhm', gain='0dB')
    i.set_frontend(2, coupling='DC', impedance='1MOhm', gain='-20dB')

    # Configure the scan oscillator to a 10 Hz 500 mVpp positive ramp
    # signal from Output 1
    i.set_scan_oscillator(enable=True, shape='PositiveRamp',
                          frequency=10, amplitude=0.5, output='Output1')

    # Configure the demodulation signal to Local oscillator with 1 MHz and
    # 0 degrees phase shift
    i.set_demodulation('Internal', frequency=1e6, phase=0)

    # Configure a 4th order low pass filter with 100 kHz corner frequency
    i.set_filter(shape='Lowpass', low_corner=100e3, order=4)

    # Set the fast PID controller to -10 dB proportional gain and
    # integrator crossover frequency at 3 kHz
    i.set_pid_by_frequency(1, -10, int_crossover=3e3)
    # Set the slow PID controller to -10 dB proportional gain and
    # integrator crossover frequency at 50 Hz
    i.set_pid_by_frequency(2, -10, int_crossover=50)


except Exception as e:
    print(f'Exception occurred: {e}')
finally:
    # Close the connection to the Moku device
    # This ensures network resources are released correctly
    i.relinquish_ownership()
