## Oscilloscope Averaging Example

## This example demonstrates how to average oscilloscope waveforms
## with the Python API. 

## (c) 2025 Liquid Instruments Pty. Ltd.


from moku.instruments import Oscilloscope
import matplotlib.pyplot as plt
import numpy as np



osc=Oscilloscope('192.168.XX.XX', force_connect=True)

try:
    ## Set frontend settings
    osc.set_frontend(channel=1, impedance='50Ohm', coupling='DC', range='4Vpp')

    ## Set channel source
    osc.set_source(channel=1, source='Input1')

    ## Set timebase and maximum number of data points
    osc.set_timebase(t1=-5e-6,t2=5e-6, max_length=1024)

    ## Set acquisition mode
    osc.set_acquisition_mode(mode='Precision')

    ## Set trigger settings
    osc.set_trigger(type='Edge', source='Input1', level=0, mode='Auto', edge='Rising')

    ## Generate waveform from Output 1
    osc.generate_waveform(channel=1, type='Sine', amplitude=1, frequency=1e6)

    ## Set empty array to store oscilloscope data,
    ## frame_averages sets the number of frames to average,
    ## data_len is the amount of points returned (adjusted with max_length)
    ## current_count keeps track of frames collected
    data_len=1024
    frame_averages=50
    data_history=np.zeros((frame_averages, data_len))
    current_count=0

    ## Initialize plot
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    line, = ax.plot([], [], 'b-', linewidth=1, label='Average')
    ax.set_xlim(-5e-6, 5e-6)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Voltage (V)')
    ax.set_title('Rolling Average')
    ax.legend()
    ax.grid(True, alpha=0.3)


    while True:

        ## Collect oscilloscope data
        data=osc.get_data()

        ## Extract time and voltage data
        t=data['time']
        v=data['ch1']

        ## Initialize circular buffer
        ## idx ranges from 1 - frame averages then resets
        ## This allows the most recent frames to be stored and averaged
        idx= current_count % frame_averages
        data_history[idx] = v
        current_count += 1

        ## This block will take the average of the last 
        ## 'frame_averages' amount of samples
        num_samples=min(current_count, frame_averages)
        averaged_signal=np.mean(data_history[:num_samples], axis=0)
    
        ## Plot the data
        line.set_data(t, averaged_signal)
        ax.set_title(f'Rolling Average (Iteration {current_count})')

        plt.draw()
        plt.pause(0.1)

        print(f'Iteration {current_count} - Using {num_samples} samples for average')

    plt.ioff()
    plt.show()

finally:
    osc.relinquish_ownership()