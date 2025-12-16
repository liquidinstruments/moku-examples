IP_ADDR = '192.168.2.200'
BITSTREAMS_PATH = 'C:/Users/heyan/Downloads/boxcarMokuPro.tar.gz'
ATTENUATION = '0dB'
INPUT_IMPEDANCE = '50Ohm'
COUPLING = 'DC'
INITIAL_NEGATIVE_TIME = -200 #ns
INITIAL_POSITIVE_TIME = 800 #ns
INITIAL_TRIGGER_LEVEL = 0.075 #Volts
INITIAL_TRIGGER_DELAY = 1440 #ns
INITIAL_GATEWIDTH = 320 #ns
INITIAL_AVERAGE_LENGTH = 100
INITIAL_OUTPUT_GAIN = 1e-2


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
from moku.instruments import Oscilloscope, CustomInstrument, MultiInstrument

print('Connecting to Moku...')
m = MultiInstrument(IP_ADDR, platform_id=4, force_connect=True)

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


mc = m.set_instrument(1, CustomInstrument, bitstream=BITSTREAMS_PATH)
osc = m.set_instrument(2, Oscilloscope)

connections = [dict(source="Input1", destination="Slot1InA"),
                dict(source="Input2", destination="Slot1InB"),
                dict(source="Slot1OutA", destination="Slot2InA"),
                dict(source="Slot1OutB", destination="Slot2InB")]
m.set_connections(connections=connections)

print(m.set_frontend(1, INPUT_IMPEDANCE, COUPLING, attenuation=ATTENUATION))
print(m.set_frontend(2, INPUT_IMPEDANCE, COUPLING, attenuation=ATTENUATION))

osc.set_trigger(type='Edge', source='ChannelB', level=0.1)
osc.set_timebase(INITIAL_NEGATIVE_TIME*ns,INITIAL_POSITIVE_TIME*ns) 
osc.set_interpolation(interpolation='Gaussian')

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

    mc.set_control(0,trg_level_bits)

    warning_text.delete("1.0", "end")
    warning_text.insert(tkinter.END, 'Coerced to ' + quantized_text + ' Volts' )

#########################################################################
def update_trg_delay(event):
    trg_delay = float(trg_delay_text.get())
    trg_delay_bits = math.ceil(trg_delay*ns/period)
    trg_delay_text.delete(0, 'end')
    quantized_text = str(trg_delay_bits*period/ns)
    print(quantized_text)
    trg_delay_text.insert(tkinter.END, quantized_text)

    mc.set_control(1,trg_delay_bits)
    
    warning_text.delete("1.0", "end")
    warning_text.insert(tkinter.END, 'Coerced to ' + quantized_text + ' ns' )

#########################################################################
def update_gate_width(event):
    gate_width = float(gate_width_text.get()) 
    gate_width_bits = math.ceil(gate_width*ns/period)
    gate_width_text.delete(0, 'end')
    quantized_text = str(gate_width_bits*period/ns)
    gate_width_text.insert(tkinter.END, quantized_text)

    mc.set_control(2,gate_width_bits)

    warning_text.delete("1.0", "end")
    warning_text.insert(tkinter.END, 'Coerced to ' + quantized_text + ' ns' )

#########################################################################
def update_avg_length(event):    
    avg_length = int(float(avg_length_text.get()))
    mc.set_control(3,avg_length)

#########################################################################
def update_gain(event):
    gain = float(gain_text.get())
    gain = int(gain*2**16)

    mc.set_control(5,gain)
    
    warning_text.delete("1.0", "end")
    warning_text.insert(tkinter.END, 'Coerced gain to ' + str(gain/2**16))

#########################################################################
def auto_delay():
    selected_option = mode.get()
    if selected_option == 'Align':
        data = osc.get_data()
        ax1.set_xlim([data['time'][0], data['time'][-1]])
        max_index = data['ch1'].index(max(data['ch1']))
        max_time = data['time'][max_index]
        gate_width = float(gate_width_text.get()) 
        trg_delay = float(trg_delay_text.get())

        time_delay = max_time - gate_width/2*ns
        time_delay_bits = math.ceil((trg_delay*ns + time_delay)/period)

        trg_delay_text.delete(0, 'end')
        trg_delay_text.insert(tkinter.END, str(time_delay_bits*period/ns))
        
        mc.set_control(1,time_delay_bits)
    else:
        return

#########################################################################
def update_mode(event):
    selected_option = mode.get()
    print(selected_option)
    match selected_option:
        case 'Align':
            output_mode = 2
            ax1.set_ylabel('Pulse Input Amplitude (Volts)', color=color_ch1)
            mc.set_control(4,output_mode)
            out_text.delete("1.0", "end")
            auto_button.configure(state = 'normal')
        case 'Average Output':
            output_mode = 0
            ax1.set_ylabel('Summed Pulse Amplitude (Volts)', color=color_ch1)
            mc.set_control(4,output_mode)
            auto_button.configure(state = 'disabled')
            

#########################################################################
## Quit the application
def quit(tk_root, instrument):
    exit_event.set()
    tk_root.destroy()
    instrument.relinquish_ownership()

#########################################################################
def update_plot():
    while True:

        if exit_event.is_set():
            break
        try:
            data = osc.get_data()
            data_time_ns = np.array(data['time'])/ns
            # print(data_time_ns)
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

            if mode.get() == 'Average Output':
                gate_width = float(gate_width_text.get()) 
                gate_width_bits = math.ceil(gate_width*ns/period)
                
                gain = int(float(gain_text.get())*2**16)/2**16
                
                out_text.delete("1.0", "end")
                warning_text.delete("1.0", "end")
                if abs(data['ch1'][0]) > range/2*saturation_threshold:
                    warning_text.insert(tkinter.END, 'Possible saturation detected, please reduce gain' )
                else:
                    avg_length = float(avg_length_text.get())
                    averaged_out = sum(data['ch1'])/(len(data['ch1'])*avg_length*mV*gain)
                    out_text.insert(tkinter.END, "{:.6f}".format(averaged_out))
                    
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
tkinter.Label(root, text="Gate Width (ns)").grid(row=4, column=matplot_columnspan+1, sticky=left_column_sticky) 
gate_width_text = tkinter.Entry(root, width = 8)
gate_width_text.insert(tkinter.END, str(INITIAL_GATEWIDTH))
gate_width_text.grid(row=4, column=matplot_columnspan+2, sticky=right_column_sticky) 
gate_width_text.bind('<Return>',update_gate_width)

## Boxcar Average Length input boxe 
tkinter.Label(root, text="# of Avg.").grid(row=3, column=matplot_columnspan+1, sticky=left_column_sticky) 
avg_length_text = tkinter.Entry(root, width = 8)
avg_length_text.insert(tkinter.END, str(INITIAL_AVERAGE_LENGTH))
avg_length_text.grid(row=3, column=matplot_columnspan+2, sticky=right_column_sticky)
avg_length_text.bind('<Return>',update_avg_length)

## Gain input box
tkinter.Label(root, text="Output Gain").grid(row=10, column=matplot_columnspan+1, sticky=left_column_sticky) 
gain_text = tkinter.Entry(root, width = 8)
gain_text.insert(tkinter.END, str(INITIAL_OUTPUT_GAIN))
gain_text.grid(row=10, column=matplot_columnspan+2, sticky=right_column_sticky) 
gain_text.bind('<Return>',update_gain)

## Trigger Delay input boxes
tkinter.Label(root, text="Pulse Trg Delay [ns]").grid(row=6, column=matplot_columnspan+1, sticky=left_column_sticky) 
trg_delay_text = tkinter.Entry(root, width = 8)
trg_delay_text.insert(tkinter.END, str(INITIAL_TRIGGER_DELAY))
trg_delay_text.grid(row=6, column=matplot_columnspan+2, sticky=right_column_sticky)
trg_delay_text.bind('<Return>',update_trg_delay)

## Trigger Delay Auto button
auto_button = tkinter.Button(root, text = "Auto",  command = auto_delay, state = 'normal')
auto_button.grid(row=6, column=matplot_columnspan+3, sticky=right_column_sticky)

## Boxcar outputs select
tkinter.Label(root, text="Mode").grid(row=7, column=matplot_columnspan+1, sticky=left_column_sticky)
mode_options = ["Align", "Average Output"]
mode = ttk.Combobox(root, values=mode_options, height = len(mode_options), width = 14)
mode.bind("<<ComboboxSelected>>", update_mode)
mode.current(0)
mode.grid(row=7, column=matplot_columnspan+2, sticky=right_column_sticky) 

## Quit button
tkinter.Button(root, text = "Quit", command = lambda: quit(root, m)).grid(row=matplot_rowspan + 2, column=matplot_columnspan+3, sticky='e')

#################################################################

## Read out
out_text = tkinter.Text(root, height = 1, width = 14)
out_text.grid(row=8, column=matplot_columnspan+2, sticky=right_column_sticky)
out_text.config(font="TkTextFont 14 normal")
tkinter.Label(root, text="Averaged Output [mV]").grid(row=8, column=matplot_columnspan+1, sticky=left_column_sticky) 

## Warning text box
warning_text = tkinter.Text(root, height = 5, width = 20, font="TkTextFont 14 normal")
warning_text.grid(row=matplot_rowspan-1, column=matplot_columnspan+1, columnspan = 3, sticky = tkinter.W+tkinter.E)

####################################################################

print('Initializing...')

mc.set_control(0,int(INITIAL_TRIGGER_LEVEL/resolution) )
mc.set_control(1,int(INITIAL_TRIGGER_DELAY*ns/period) )
mc.set_control(2,int(INITIAL_GATEWIDTH*ns/period) )
mc.set_control(3,int(INITIAL_AVERAGE_LENGTH) )
mc.set_control(4,INITIAL_OUTPUT_MODE) 
mc.set_control(5,int(INITIAL_OUTPUT_GAIN*2**16) )

ax1.set_ylabel('Pulse Input Amplitude (Volts)', color=color_ch1)
    
#########################################################################

## Matplotlib box
canvas.get_tk_widget().grid(row=0, column=0, rowspan=60, columnspan=120, ipadx=100, ipady=20)
tkinter.mainloop()


