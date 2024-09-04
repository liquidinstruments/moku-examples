#
# moku example: Phasemeter networking streaming
#
# This example starts a 30-second network stream of Channel 1 Phasemeter data
# and processes it live. The contents of each data sample are printed out,
# along with the signal amplitude which may be calculated as A = sqrt(I^2 + Q^2).
# 
# (c) 2024 Liquid Instruments Pty. Ltd.
#

from moku.instruments import Phasemeter
import numpy as np
import matplotlib.pyplot as plt

# Connect to your Moku by its ip address using Phasemeter('192.168.###.###')
# or by its serial number using Phasemeter(serial=123)
i = Phasemeter('192.168.###.###', force_connect=False)

try:
    # Set samplerate to 150 Hz/s
    i.set_acquisition_speed(speed='150Hz')

    # Set channel 1 to DC coupled, 1 MOhm impedance, and 4Vpp range
    i.set_frontend(1, coupling='DC', impedance='1MOhm', range='4Vpp')

    # Set auto acquired frequency for channel 1
    i.set_pm_loop(channel=1, auto_acquire=True, bandwidth='1kHz')

    # Get auto acquired frequency for channel 1
    i.get_auto_acquired_frequency(channel=1)

    # Stop and existing streaming session and start a new one for 30s
    i.stop_streaming()
    i.start_streaming(duration=30)

    # Set up the figure for plotting
    plt.ion()
    fig, axs = plt.subplots(3, 1, sharex=True)
    plt.show()
    for ax in axs:
        ax.grid(True)
        ax.set_xlim(0, 30)
    (f_ax, a_ax, p_ax) = axs

    f_ax.set_ylabel("Frequency (Hz)")
    a_ax.set_ylabel("Amplitude (Vrms)")
    p_ax.set_ylabel("Phase (cyc)")
    p_ax.set_xlabel("Time (s)")
    
    (line1,) = f_ax.plot([])
    (line2,) = f_ax.plot([])
    (line3,) = a_ax.plot([])
    (line4,) = p_ax.plot([])
    f_ax.legend(labels=("Frequency", "Set Frequency"), loc=1)

    time_span = []
    frequency = []
    set_freq_ = []
    amplitude = []
    phase_cyc = []

    # This loop continuously updates the plot with new data
    while True:
        # Get stream data
        data = i.get_stream_data()

        # Update the plot
        time_span += data['time']
        frequency += data['ch1_frequency']
        set_freq_ += data['ch1_set_frequency']
        amplitude += list(np.sqrt(np.array(data['ch1_i'])**2+np.array(data['ch1_q'])**2))
        phase_cyc += data['ch1_phase']
        line1.set_data(time_span, frequency)
        line2.set_data(time_span, set_freq_)
        line3.set_data(time_span, amplitude)
        line4.set_data(time_span, phase_cyc)
        for ax in axs:
            ax.relim()
            ax.autoscale_view()
        plt.pause(0.01)
  
except Exception as e:
    print(f'Exception Occurred: {e}')
    plt.pause(2)
finally:
    # Close the connection to the Moku device
    # This ensures network resources are released correctly
    i.stop_streaming()
    i.relinquish_ownership()