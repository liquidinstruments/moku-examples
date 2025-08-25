IP_ADDR = '192.168.2.211'
BITSTREAMS_PATH = 'C:/Users/heyan/Downloads/FourBoxcarAveragerMokuProf_fw591.tar.gz'
ATTENUATION = '0dB'
INPUT_IMPEDANCE = '50Ohm'
COUPLING = 'DC'
INITIAL_NEGATIVE_TIME = -100 #ns
INITIAL_POSITIVE_TIME = 500 #ns
INITIAL_TRIGGER_LEVEL = 0.075 #Volts
INITIAL_TRIGGER_DELAY = 184 #ns
INITIAL_BASELINE_TRIGGER_DELAY = 512 #ns
INITIAL_GATEWIDTH = 100 #ns
INITIAL_AVERAGE_LENGTH = 100
INITIAL_OUTPUT_GAIN = 1e-2
NUM_OF_SLOTS = 4 # Moku:Pro = 4, Moku:Lab = 2, Moku:Go = 2

######################################
INITIAL_OUTPUT_MODE= 2
INITIAL_OUTPUT_SELECTION = 0

ns = 1e-9
mV = 1e-3


import tkinter
from tkinter import ttk
import time
from threading import *
import math
import numpy as np

from matplotlib.backend_bases import key_press_handler
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
from matplotlib.figure import Figure
from moku.instruments import Oscilloscope, CloudCompile, MultiInstrument

print('Connecting to Moku...')
m = MultiInstrument(IP_ADDR, platform_id=NUM_OF_SLOTS, force_connect=True)

print('Setting Multi-instrument Mode...')
description = m.describe()

period_dict = {
    'Moku:Go': 32e-9,
    'Moku:Lab': 8e-9,
    'Moku:Pro': 3.2e-9,
    'Moku:Delta': 3.2e-9
}

resolution_dict = {
    'Moku:Go': 1/6550.4, # 6550.4 bits/volt
    'Moku:Lab': 2/30000, # 30000  bits/volt
    'Moku:Pro': 1/29925,  # 29925  bits/volt
    'Moku:Delta': 1/36440  # 36440  bits/volt
}

range_dict ={
    'Moku:Go': 10,
    'Moku:Lab': 2,
    'Moku:Pro': 2,
    'Moku:Delta': 1
}

saturation_threshold = 0.95

period = period_dict[description['hardware']]
resolution = resolution_dict[description['hardware']]
range = range_dict[description['hardware']]


mcc = m.set_instrument(1, CloudCompile, bitstream=BITSTREAMS_PATH)
osc = m.set_instrument(2, Oscilloscope)

connections = [dict(source="Input1", destination="Slot1InA"),
                dict(source="Input2", destination="Slot1InB"),
                dict(source="Slot1OutA", destination="Slot2InA"),
                dict(source="Slot1OutB", destination="Slot2InB")]
m.set_connections(connections=connections)

print(m.set_frontend(1, INPUT_IMPEDANCE, COUPLING, ATTENUATION))
print(m.set_frontend(2, INPUT_IMPEDANCE, COUPLING, ATTENUATION))

osc.set_trigger(type='Edge', source='ChannelB', level=0.4)
osc.set_timebase(INITIAL_NEGATIVE_TIME*ns,INITIAL_POSITIVE_TIME*ns) 
osc.set_interpolation('Gaussian')

fig = Figure(figsize=(5, 4), dpi=100)
ax1 = fig.add_subplot()
data = osc.get_data()
color_ch1 = 'tab:blue'
data_time_ns = np.array(data['time'])/ns

line1, = ax1.plot(data_time_ns, data['ch1'],color=color_ch1)
ax1.set_xlim([data_time_ns[0], data_time_ns[-1]])
ax1.set_xlabel("time [ns]")
ax1.set_ylabel('Ch1 Amplitude (Volts)', color=color_ch1)
ax1.tick_params(axis='y', labelcolor=color_ch1)

ax2 = ax1.twinx()  
color_ch2 = 'tab:red'
ax2.set_ylabel('Boxcar Window (Volts)', color=color_ch2)  # we already handled the x-label with ax1
line2, = ax2.plot(data_time_ns, data['ch2'],color=color_ch2)
ax2.tick_params(axis='y', labelcolor=color_ch2)
ax2.set_xlim([data_time_ns[0], data_time_ns[-1]])

#Need to leave some space around the plot. When we change to average the tick labels are very long
#Might be better to fix this using a format string for the tick labels
fig.subplots_adjust(left=0.15, bottom=0.15, right=0.85, top=0.95, wspace=0, hspace=0)

left_column_sticky = 'e'
right_column_sticky = 'w'

root = tkinter.Tk()

canvas = FigureCanvasTkAgg(fig, master=root)  # A tk.DrawingArea.
canvas.draw()

####################################################################
def enable_averager_0(enable):
    if enable == True:
        trg_delay_text_0.config(state='normal')
        trg_delay_text_baseline_0.config(state='normal')
    else:
        trg_delay_text_0.config(state='disabled')
        trg_delay_text_baseline_0.config(state='disabled')

def enable_averager_1(enable):
    if enable == True:
        trg_delay_text_1.config(state='normal')
        trg_delay_text_baseline_1.config(state='normal')
    else:
        trg_delay_text_1.config(state='disabled')
        trg_delay_text_baseline_1.config(state='disabled')
        
def enable_gain_0(enable):       
    if enable ==True:
        gain_text_0.config(state='normal')
    else:
        gain_text_0.config(state='disabled')
        
def enable_gain_1(enable):       
    if enable ==True:
        gain_text_1.config(state='normal')
    else:
        gain_text_1.config(state='disabled')
#########################################################################
def update_timebase(event):
    neg_timebase = float(neg_timebase_text.get()) 
    pos_timebase = float(pos_timebase_text.get()) 
    print(osc.set_timebase(neg_timebase*ns, pos_timebase*ns)) 

#########################################################################
def update_trg_level(event):
    trg_level = float(trg_level_text.get())
    trg_level_bits = math.ceil(trg_level/resolution)

    trg_level_text.delete(0, 'end')
    quantized_text = str(trg_level_bits*resolution)
    trg_level_text.insert(tkinter.END, quantized_text)

    print(mcc.set_control(0,trg_level_bits))

    warning_text.delete("1.0", "end")
    warning_text.insert(tkinter.END, 'Coerced to ' + quantized_text + ' Volts' )

#########################################################################
def update_gate_width(event):
    gate_width = float(gate_width_text.get()) 
    gate_width_bits = math.ceil(gate_width*ns/period)
    gate_width_text.delete(0, 'end')
    quantized_text = str(gate_width_bits*period/ns)
    gate_width_text.insert(tkinter.END, quantized_text)

    print(mcc.set_control(3,gate_width_bits))
    print(mcc.set_control(8,gate_width_bits))

    warning_text.delete("1.0", "end")
    warning_text.insert(tkinter.END, 'Coerced to ' + quantized_text + ' ns' )

#########################################################################
def update_avg_length(event):    
    avg_length = int(float(avg_length_text.get()))
    
    print(mcc.set_control(4,avg_length))
    print(mcc.set_control(9,avg_length))
    
    warning_text.delete("1.0", "end")

#########################################################################
def update_gain_0(event):
    gain = float(gain_text_0.get())
    gain = int(gain*2**16)

    print(mcc.set_control(5,gain))
    
    warning_text.delete("1.0", "end")
    warning_text.insert(tkinter.END, 'Coerced gain to ' + str(gain/2**16))
    
#########################################################################
def update_gain_1(event):
    gain = float(gain_text_1.get())
    gain = int(gain*2**16)

    print(mcc.set_control(10,gain))
    
    warning_text.delete("1.0", "end")
    warning_text.insert(tkinter.END, 'Coerced gain to ' + str(gain/2**16))

#########################################################################
def update_trg_delay_0(event):
    trg_delay = float(trg_delay_text_0.get())
    trg_delay_bits = math.ceil(trg_delay*ns/period)
    trg_delay_text_0.delete(0, 'end')
    quantized_text = str(trg_delay_bits*period/ns)
    print(quantized_text)
    trg_delay_text_0.insert(tkinter.END, quantized_text)

    print(mcc.set_control(1,trg_delay_bits))
    
    warning_text.delete("1.0", "end")
    warning_text.insert(tkinter.END, 'Coerced to ' + quantized_text + ' ns' )
    
#########################################################################
def update_trg_delay_1(event):
    trg_delay = float(trg_delay_text_1.get())
    trg_delay_bits = math.ceil(trg_delay*ns/period)
    trg_delay_text_1.delete(0, 'end')
    quantized_text = str(trg_delay_bits*period/ns)
    print(quantized_text)
    trg_delay_text_1.insert(tkinter.END, quantized_text)

    print(mcc.set_control(6,trg_delay_bits))
    
    warning_text.delete("1.0", "end")
    warning_text.insert(tkinter.END, 'Coerced to ' + quantized_text + ' ns' )    

############################################################################
def update_trg_delay_baseline_0(event):
    trg_delay = float(trg_delay_text_baseline_0.get())

    trg_delay_bits = math.ceil(trg_delay*ns/period)
    trg_delay_text_baseline_0.delete(0, 'end')
    quantized_text = str(trg_delay_bits*period/ns)
    print(quantized_text)
    trg_delay_text_baseline_0.insert(tkinter.END, quantized_text)

    print(mcc.set_control(2,trg_delay_bits))

    warning_text.delete("1.0", "end")
    warning_text.insert(tkinter.END, 'Coerced to ' + quantized_text + ' ns' )

###########################################################################
def update_trg_delay_baseline_1(event):
    trg_delay = float(trg_delay_text_baseline_1.get())

    trg_delay_bits = math.ceil(trg_delay*ns/period)
    trg_delay_text_baseline_1.delete(0, 'end')
    quantized_text = str(trg_delay_bits*period/ns)
    print(quantized_text)
    trg_delay_text_baseline_1.insert(tkinter.END, quantized_text)

    print(mcc.set_control(7,trg_delay_bits)) 

    warning_text.delete("1.0", "end")
    warning_text.insert(tkinter.END, 'Coerced to ' + quantized_text + ' ns' )

#########################################################################
def update_mode(event):
    selected_option = mode.get()
    print(selected_option)
    # auto_button.config(state='disabled')
    match selected_option:
        case 'Align Averager0':
            enable_averager_0(True)
            enable_averager_1(False)
            enable_gain_0(False)
            enable_gain_1(False)
            ax1.set_ylabel('Pulse Input Amplitude (Volts)', color=color_ch1)
            print(mcc.set_control(15,15))
            out_text_0.delete("1.0", "end")
            out_text_1.delete("1.0", "end")

        case 'Output Averager0':
            enable_averager_0(False)
            enable_averager_1(False)
            enable_gain_0(True)
            enable_gain_1(False)
            ax1.set_ylabel('Summed Pulse Amplitude_0 (Volts)', color=color_ch1)
            print(mcc.set_control(15,7)) 

        case 'Align Averager1':
            enable_averager_0(False)
            enable_averager_1(True)
            enable_gain_0(False)
            enable_gain_1(False)
            ax1.set_ylabel('Pulse Input Amplitude (Volts)', color=color_ch1)
            print(mcc.set_control(15,13))
            out_text_0.delete("1.0", "end")
            out_text_1.delete("1.0", "end")
            
        case 'Output Averager1':
            enable_averager_0(False)
            enable_averager_1(False)
            enable_gain_0(False)
            enable_gain_1(True)
            ax1.set_ylabel('Summed Pulse Amplitude_1 (Volts)', color=color_ch1)
            print(mcc.set_control(15,9))

        case 'Output Both':
            enable_averager_0(False)
            enable_averager_1(False)
            enable_gain_0(True)
            enable_gain_1(True)
            ax1.set_ylabel('Summed Pulse Amplitude_0 (Volts)', color=color_ch1)
            ax2.set_ylabel('Summed Pulse Amplitude_1 (Volts)', color=color_ch2)
            print(mcc.set_control(15,4))
            

#########################################################################
## Quit the application
def quit(tk_root, instrument):
    exit_event.set()
    tk_root.destroy()
    print(mcc.get_controls())
    instrument.relinquish_ownership()

#########################################################################
def update_plot():
    while True:

        if exit_event.is_set():
            break
        try:
            data = osc.get_data()
            data_time_ns = np.array(data['time'])/ns
            ax1.set_xlim([data_time_ns[0], data_time_ns[-1]])
            
            min_ch1 = min(data['ch1'])
            max_ch1 = max(data['ch1'])
            ax1.set_ylim([min_ch1-0.001, max_ch1+0.001])

            min_ch2 = min(data['ch2'])
            max_ch2 = max(data['ch2'])
            ax2.set_ylim([min_ch2-0.001, max_ch2+0.1])

            line1.set_data(data_time_ns, data['ch1'])
            line2.set_data(data_time_ns, data['ch2'])
            canvas.draw()

            if mode.get() == 'Output Averager0':
                gate_width = float(gate_width_text.get()) 
                gate_width_bits = math.ceil(gate_width*ns/period)
                
                gain = int(float(gain_text_0.get())*2**16)/2**16
                
                out_text_0.delete("1.0", "end")
                out_text_1.delete("1.0", "end")
                warning_text.delete("1.0", "end")
                if abs(data['ch1'][0]) > range/2*saturation_threshold:
                    warning_text.insert(tkinter.END, 'Possible saturation detected, please reduce gain' )
                else:
                    avg_length = float(avg_length_text.get())
                    averaged_out = sum(data['ch1'])/(len(data['ch1'])*avg_length*mV*gain)
                    out_text_0.insert(tkinter.END, "{:.6f}".format(averaged_out))
                
                
            elif mode.get() == 'Output Averager1':
                gate_width = float(gate_width_text.get()) 
                gate_width_bits = math.ceil(gate_width*ns/period)
                
                gain = int(float(gain_text_1.get())*2**16)/2**16
                
                out_text_0.delete("1.0", "end")
                out_text_1.delete("1.0", "end")
                warning_text.delete("1.0", "end")
                if abs(data['ch1'][0]) > range/2*saturation_threshold:
                    warning_text.insert(tkinter.END, 'Possible saturation detected, please reduce gain' )
                else:
                    avg_length = float(avg_length_text.get())
                    averaged_out = sum(data['ch1'])/(len(data['ch1'])*avg_length*mV*gain)
                    out_text_1.insert(tkinter.END, "{:.6f}".format(averaged_out))
                
                
            elif mode.get() == 'Output Both':
                gate_width = float(gate_width_text.get()) 
                gate_width_bits = math.ceil(gate_width*ns/period)
                
                gain_0 = int(float(gain_text_0.get())*2**16)/2**16
                gain_1 = int(float(gain_text_1.get())*2**16)/2**16

                out_text_0.delete("1.0", "end")
                out_text_1.delete("1.0", "end")
                warning_text.delete("1.0", "end")
                if abs(data['ch1'][0]) > range/2*saturation_threshold or abs(data['ch2'][0]) > range/2*saturation_threshold:
                    warning_text.insert(tkinter.END, 'Possible saturation detected, please reduce gain' )
                else:
                    avg_length = float(avg_length_text.get())
                    
                    averaged_out_0 = sum(data['ch1'])/(len(data['ch1'])*avg_length*mV*gain_0)
                    averaged_out_1 = sum(data['ch2'])/(len(data['ch2'])*avg_length*mV*gain_1)
                    out_text_0.insert(tkinter.END, "{:.6f}".format(averaged_out_0))
                    out_text_1.insert(tkinter.END, "{:.6f}".format(averaged_out_1))
            
            elif mode.get() == 'Align Averager0' or mode.get() == 'Align Averager1':
                warning_text.delete("1.0", "end")
                if max(data['ch2']) > (2**14+2**10)*resolution:
                    warning_text.insert(tkinter.END, 'Two gate windows are overlapping, please adjust trigger offsets' )
                    
                    
        except Exception as e:
            print(f'Exception occurred: {e}')
            continue
        time.sleep(0.4)

t1=Thread(target=update_plot)
exit_event = Event()
t1.start() 

print('Waiting for inputs...')

#########################################################################

matplot_rowspan = 60
matplot_columnspan = 120

## Timebase input boxes 
tkinter.Label(root, text="Neg (ns)").grid(row=matplot_rowspan + 1, column=3) 
neg_timebase_text = tkinter.Entry(root, width = 8)
neg_timebase_text.insert(tkinter.END, str(INITIAL_NEGATIVE_TIME))
neg_timebase_text.grid(row=matplot_rowspan + 2, column=3) 
neg_timebase_text.bind('<Return>',update_timebase)

tkinter.Label(root, text="Pos (ns)").grid(row=matplot_rowspan + 1, column=matplot_columnspan-3) 
pos_timebase_text = tkinter.Entry(root, width = 8)
pos_timebase_text.insert(tkinter.END, str(INITIAL_POSITIVE_TIME))
pos_timebase_text.grid(row=matplot_rowspan + 2, column=matplot_columnspan-3)
pos_timebase_text.bind('<Return>',update_timebase)

## Trigger Level input boxes
tkinter.Label(root, text="Trigger Level (Volts)").grid(row=2, column=matplot_columnspan+1, sticky=left_column_sticky) 
trg_level_text = tkinter.Entry(root, width = 8)
trg_level_text.insert(tkinter.END, str(INITIAL_TRIGGER_LEVEL))
trg_level_text.grid(row=2, column=matplot_columnspan+2, sticky = right_column_sticky) 
trg_level_text.bind('<Return>',update_trg_level)

## Boxcar Gate Width input boxe
tkinter.Label(root, text="Gate Width (ns)").grid(row=3, column=matplot_columnspan+1, sticky=left_column_sticky) 
gate_width_text = tkinter.Entry(root, width = 8)
gate_width_text.insert(tkinter.END, str(INITIAL_GATEWIDTH))
gate_width_text.grid(row=3, column=matplot_columnspan+2, sticky=right_column_sticky) 
gate_width_text.bind('<Return>',update_gate_width)

## Boxcar Average Length input boxe 
tkinter.Label(root, text="# of Avg.").grid(row=4, column=matplot_columnspan+1, sticky=left_column_sticky) 
avg_length_text = tkinter.Entry(root, width = 8)
avg_length_text.insert(tkinter.END, str(INITIAL_AVERAGE_LENGTH))
avg_length_text.grid(row=4, column=matplot_columnspan+2, sticky=right_column_sticky)
avg_length_text.bind('<Return>',update_avg_length)

## Trigger Delay_0 input boxes
tkinter.Label(root, text="Pulse Trg Delay_0 [ns]").grid(row=5, column=matplot_columnspan+1, sticky=left_column_sticky) 
trg_delay_text_0 = tkinter.Entry(root, width = 8)
trg_delay_text_0.insert(tkinter.END, str(INITIAL_TRIGGER_DELAY))
trg_delay_text_0.grid(row=5, column=matplot_columnspan+2, sticky=right_column_sticky)
trg_delay_text_0.bind('<Return>',update_trg_delay_0)

## Baseline Trigger_0 Delay input boxes
tkinter.Label(root, text="Baseline Trg Delay_0 [ns]").grid(row=6, column=matplot_columnspan+1, sticky=left_column_sticky) 
trg_delay_text_baseline_0 = tkinter.Entry(root, width = 8)
trg_delay_text_baseline_0.insert(tkinter.END, str(INITIAL_BASELINE_TRIGGER_DELAY))
trg_delay_text_baseline_0.grid(row=6, column=matplot_columnspan+2, sticky=right_column_sticky)
trg_delay_text_baseline_0.bind('<Return>',update_trg_delay_baseline_0)

## Gain_0 input box
tkinter.Label(root, text="Output Gain_0").grid(row=7, column=matplot_columnspan+1, sticky=left_column_sticky) 
gain_text_0 = tkinter.Entry(root, width = 8)
gain_text_0.insert(tkinter.END, str(INITIAL_OUTPUT_GAIN))
gain_text_0.grid(row=7, column=matplot_columnspan+2, sticky=right_column_sticky) 
gain_text_0.bind('<Return>',update_gain_0)

## Trigger Delay_1 input boxes
tkinter.Label(root, text="Pulse Trg Delay_1 [ns]").grid(row=8, column=matplot_columnspan+1, sticky=left_column_sticky) 
trg_delay_text_1 = tkinter.Entry(root, width = 8)
trg_delay_text_1.insert(tkinter.END, str(INITIAL_TRIGGER_DELAY))
trg_delay_text_1.grid(row=8, column=matplot_columnspan+2, sticky=right_column_sticky)
trg_delay_text_1.bind('<Return>',update_trg_delay_1)

## Baseline Trigger Delay_1 input boxes
tkinter.Label(root, text="Baseline Trg Delay_1 [ns]").grid(row=9, column=matplot_columnspan+1, sticky=left_column_sticky) 
trg_delay_text_baseline_1 = tkinter.Entry(root, width = 8)
trg_delay_text_baseline_1.insert(tkinter.END, str(INITIAL_BASELINE_TRIGGER_DELAY))
trg_delay_text_baseline_1.grid(row=9, column=matplot_columnspan+2, sticky=right_column_sticky)
trg_delay_text_baseline_1.bind('<Return>',update_trg_delay_baseline_1)

## Gain_1 input box
tkinter.Label(root, text="Output Gain_1").grid(row=10, column=matplot_columnspan+1, sticky=left_column_sticky) 
gain_text_1 = tkinter.Entry(root, width = 8)
gain_text_1.insert(tkinter.END, str(INITIAL_OUTPUT_GAIN))
gain_text_1.grid(row=10, column=matplot_columnspan+2, sticky=right_column_sticky) 
gain_text_1.bind('<Return>',update_gain_1)

## Boxcar outputs select
tkinter.Label(root, text="Mode").grid(row=11, column=matplot_columnspan+1, sticky=left_column_sticky)
mode_options = ["Align Averager0", "Output Averager0", "Align Averager1", "Output Averager1", "Output Both"]
mode = ttk.Combobox(root, values=mode_options, height = len(mode_options), width = 16)
mode.bind("<<ComboboxSelected>>", update_mode)
mode.current(0)
mode.grid(row=11, column=matplot_columnspan+2, sticky=right_column_sticky) 

## Read out_0
tkinter.Label(root, text="Averaged Output_0 [mV]").grid(row=12, column=matplot_columnspan+1, sticky=left_column_sticky) 
out_text_0 = tkinter.Text(root, height = 1, width = 14)
out_text_0.grid(row=12, column=matplot_columnspan+2, sticky=right_column_sticky)
out_text_0.config(font="TkTextFont 14 normal")

## Read out_1
tkinter.Label(root, text="Averaged Output_1 [mV]").grid(row=13, column=matplot_columnspan+1, sticky=left_column_sticky) 
out_text_1 = tkinter.Text(root, height = 1, width = 14)
out_text_1.grid(row=13, column=matplot_columnspan+2, sticky=right_column_sticky)
out_text_1.config(font="TkTextFont 14 normal")

## Warning text box
warning_text = tkinter.Text(root, height = 5, width = 20, font="TkTextFont 14 normal")
warning_text.grid(row=matplot_rowspan-1, column=matplot_columnspan+1, columnspan = 3, sticky = tkinter.W+tkinter.E)

## Quit button
tkinter.Button(root, text = "Quit", command = lambda: quit(root, m)).grid(row=matplot_rowspan + 2, column=matplot_columnspan+3, sticky='e')

####################################################################
print('Initializing...')

mcc.set_control(0,int(INITIAL_TRIGGER_LEVEL/resolution) )

mcc.set_control(1,int(INITIAL_TRIGGER_DELAY*ns/period) )
mcc.set_control(2,int(INITIAL_BASELINE_TRIGGER_DELAY*ns/period))

mcc.set_control(6,int(INITIAL_TRIGGER_DELAY*ns/period) )
mcc.set_control(7,int(INITIAL_BASELINE_TRIGGER_DELAY*ns/period))

mcc.set_control(3,int(INITIAL_GATEWIDTH*ns/period) )
mcc.set_control(8,int(INITIAL_GATEWIDTH*ns/period) )

mcc.set_control(4,int(INITIAL_AVERAGE_LENGTH) )
mcc.set_control(9,int(INITIAL_AVERAGE_LENGTH) )

mcc.set_control(5,int(INITIAL_OUTPUT_GAIN*2**16) )
mcc.set_control(10,int(INITIAL_OUTPUT_GAIN*2**16) )

mcc.set_control(15,INITIAL_OUTPUT_SELECTION) 

mcc.set_control(15,15)

ax1.set_ylabel('Pulse Input Amplitude (Volts)', color=color_ch1)

enable_averager_0(True)
enable_averager_1(False)
enable_gain_0(False)
enable_gain_1(False)
    
#########################################################################

## Matplotlib box
canvas.get_tk_widget().grid(row=0, column=0, rowspan=60, columnspan=120, ipadx=100, ipady=20)
tkinter.mainloop()


