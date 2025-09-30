## Oscilloscope Averaging Example

## This example demonstrates how to average oscilloscope waveforms
## with the Python API. 

## (c) 2025 Liquid Instruments Pty. Ltd.


from moku.instruments import Oscilloscope
import matplotlib.pyplot as plt



osc=Oscilloscope('192.168.XX.XX', force_connect=True)

try:
    ## Set frontend settings
    osc.set_frontend(channel=1, impedance='50Ohm', coupling='DC', range='4Vpp')

    ## Set channel source
    osc.set_source(channel=1, source='Input1')

    ## Set timebase 
    osc.set_timebase(t1=-5e-6,t2=5e-6)

    ## Set acquisition mode
    osc.set_acquisition_mode(mode='Precision')

    ## Set trigger settings
    osc.set_trigger(type='Edge', source='Input1', level=0, mode='Auto', edge='Rising')

    ## Generate waveform from Output 1
    osc.generate_waveform(channel=1, type='Sine', amplitude=1, frequency=1e6)

    ##Set the number of averages
    averages=20

    ## This initiates an empty array for the frames of data to be collected
    total_data=[]

    ## This loop sets up the data collection. Many data frames will be collected
    ## to be averaged.
    for i in range(averages):
        data=osc.get_data()
        total_data.append(data['ch1'])

    ## This adds all of the data column wise
    total=[sum(col) for col in zip(*total_data)]

    ##This line calculates the average
    average=[j/averages for j in total]

    ## Plots the results
    plt.plot(data['time'], average)
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (V)')
    plt.show()


except Exception as e:

    osc.relinquish_ownership()
    raise e

finally:
    osc.relinquish_ownership()



