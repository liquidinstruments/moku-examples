ATTENUATION = '-20dB'
INPUT_IMPEDANCE = '50Ohm'
COUPLING = 'DC'
INITIAL_NEGATIVE_TIME = -5000 #ns
INITIAL_POSITIVE_TIME = 5000 #ns
INITIAL_TRIGGER_LEVEL = 0.01 #Volts
INITIAL_TRIGGER_DELAY = 1440 #ns
INITIAL_GATEWIDTH = 320 #ns
INITIAL_GATE_SWEEP_WIN = 2*INITIAL_GATEWIDTH #ns
INITIAL_AVERAGE_LENGTH = 100
INITIAL_OUTPUT_GAIN = 1e-2
BITSTREAMS_PATH = r"C:\Users\Laura\Downloads\boxcar_mokupro_fw600.tar"

######################################
INITIAL_OUTPUT_MODE= 2
INITIAL_OUTPUT_SELECTION = 0
line3, line4 = None, None
gate_sweep_window_label = None
gate_sweep_window_text = None


from moku.instruments import MultiInstrument, WaveformGenerator, ArbitraryWaveformGenerator, CloudCompile, Oscilloscope
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import tkinter
from tkinter import ttk
from threading import *
from matplotlib.backend_bases import key_press_handler
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
from matplotlib.figure import Figure

print('Connecting to Moku...')
m = MultiInstrument('192.168.2.200', force_connect=True,platform_id=4)

print('Setting Multi-instrument Mode...')
description = m.describe()

period_dict = {
    'Moku:Go': 32e-9,
    'Moku:Lab': 8e-9,
    'Moku:Pro': 3.2e-9
}

resolution_dict = {
    'Moku:Go': 1/6550.4, # 6550.4 bits/volt
    'Moku:Lab': 2/30000, # 30000  bits/volt
    'Moku:Pro': 1/29925  # 29925  bits/volt
}

range_dict ={
    'Moku:Go': 10,
    'Moku:Lab': 2,
    'Moku:Pro': 2
}

saturation_threshold = 0.95

period = period_dict[description['hardware']]
resolution = resolution_dict[description['hardware']]
range_hardware = range_dict[description['hardware']]

ns = 1e-9 # Define ns scaling
mV = 1e-3

# Set connections for MiM
mcc = m.set_instrument(1, CloudCompile, bitstream=BITSTREAMS_PATH)
osc = m.set_instrument(2, Oscilloscope)

connections = [dict(source="Input1", destination="Slot1InA"),
                dict(source="Input2", destination="Slot1InB"),
                dict(source="Slot1OutA", destination="Slot2InA"),
                dict(source="Slot1OutB", destination="Slot2InB")]
m.set_connections(connections=connections)

print(m.set_frontend(1, INPUT_IMPEDANCE, COUPLING, ATTENUATION))
print(m.set_frontend(2, INPUT_IMPEDANCE, COUPLING, ATTENUATION))

osc.set_trigger(type='Edge', source='ChannelB', level=0.1)
osc.set_timebase(INITIAL_NEGATIVE_TIME*ns,INITIAL_POSITIVE_TIME*ns) 
osc.set_interpolation(interpolation='Gaussian')

# Set up the plotting parameters
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

    mcc.set_control(0,trg_level_bits)

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

    mcc.set_control(1,trg_delay_bits)
    
    warning_text.delete("1.0", "end")
    warning_text.insert(tkinter.END, 'Coerced to ' + quantized_text + ' ns' )

#########################################################################
def update_gate_width(event):
    gate_width = float(gate_width_text.get()) 
    gate_width_bits = math.ceil(gate_width*ns/period)
    gate_width_text.delete(0, 'end')
    quantized_text = str(gate_width_bits*period/ns)
    gate_width_text.insert(tkinter.END, quantized_text)

    mcc.set_control(2,gate_width_bits)
    update_green_pulse(None)

    warning_text.delete("1.0", "end")
    warning_text.insert(tkinter.END, 'Coerced to ' + quantized_text + ' ns' )

#########################################################################
def update_avg_length(event):    
    avg_length = int(float(avg_length_text.get()))
    mcc.set_control(3,avg_length)

#########################################################################
def update_gain(event):
    gain = float(gain_text.get())
    gain = int(gain*2**16)

    mcc.set_control(5,gain)
    
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
        
        mcc.set_control(1,time_delay_bits)
    else:
        return

#########################################################################
def update_mode(event):
    global last_mode, last_boxcar_mode

    selected_option = mode.get()
    selected_boxcar_mode = bc_mode.get()
    #print(f"Mode: {selected_option}, Boxcar Mode: {selected_boxcar_mode}")

    # Clear plot if switching from average output recovery waveform back to Moku channel 1 and 2 input waveforms
    if last_boxcar_mode == "Waveform Recovery" and last_mode == "Average Output" and (
        selected_boxcar_mode != "Waveform Recovery" or selected_option != "Average Output"
    ):
        ax1.cla()
        ax2.cla()
        canvas.draw()
    elif ( # Clear plot if switching from Moku channel 1 and 2 input waveforms to average output recovery waveform
        last_mode != "Average Output"
        and last_boxcar_mode != "Waveform Recovery"
        and selected_boxcar_mode == "Waveform Recovery"
        and selected_option == "Average Output"
    ):
        ax1.cla()
        ax2.cla()
        canvas.draw()

    # Update state for the current mode and boxcar mode
    last_mode = selected_option
    last_boxcar_mode = selected_boxcar_mode

    # Default behavior for Align mode (applies to both Static Gate and Waveform Recovery)
    if selected_option == "Align":
        output_mode = 2
        ax1.set_ylabel('Pulse Input Amplitude (Volts)', color=color_ch1)
        mcc.set_control(4, output_mode)
        out_text.delete("1.0", "end")
        auto_button.configure(state='normal')
        trg_delay_text.configure(state='normal')
        gate_width_text.configure(state='normal')
        trg_level_text.configure(state='normal')
        avg_length_text.configure(state='normal')
        neg_timebase_text.configure(state='normal')
        pos_timebase_text.configure(state='normal')

    # Specific behavior for Average Output in Waveform Recovery
    elif selected_option == "Average Output" and bc_mode.get() == "Waveform Recovery":
        output_mode = 0
        ax1.set_ylabel('Average Output Amplitude (Volts)', color=color_ch1)
        mcc.set_control(4, output_mode)
        auto_button.configure(state='disabled')
        trg_delay_text.configure(state='disabled')
        gate_width_text.configure(state='disabled')
        trg_level_text.configure(state='disabled')
        avg_length_text.configure(state='disabled')
        neg_timebase_text.configure(state='disabled')
        pos_timebase_text.configure(state='disabled')

    # Default behavior for Average Output in Static Gate
    elif selected_option == "Average Output":
        output_mode = 0
        ax1.set_ylabel('Summed Pulse Amplitude (Volts)', color=color_ch1)
        mcc.set_control(4, output_mode)
        auto_button.configure(state='disabled')
        trg_delay_text.configure(state='normal')
        gate_width_text.configure(state='normal')
        trg_level_text.configure(state='normal')
        avg_length_text.configure(state='normal')
        neg_timebase_text.configure(state='normal')
        pos_timebase_text.configure(state='normal')

last_mode = "Align"
last_boxcar_mode = "Static Gate"  
matplot_rowspan = 60
matplot_columnspan = 120 
## Boxcar outputs select
tkinter.Label(root, text="Mode").grid(row=8, column=matplot_columnspan+1, sticky=left_column_sticky)
mode_options = ["Align", "Average Output"]
mode = ttk.Combobox(root, values=mode_options, height = len(mode_options), width = 14)
mode.bind("<<ComboboxSelected>>", update_mode)
mode.set("Align")
mode.grid(row=8, column=matplot_columnspan+2, sticky=right_column_sticky) 
#########################################################################

def update_boxcar_mode(event):
    global gate_sweep_window_label, gate_sweep_window_text, line3

    selected_option = bc_mode.get()
    #print(selected_option)
    match selected_option:
        case 'Static Gate':
            output_mode = 2
            ax1.set_ylabel('Pulse Input Amplitude (Volts)', color=color_ch1)
            mcc.set_control(4, output_mode)
            out_text.delete("1.0", "end")
            auto_button.configure(state='normal')

            # Remove "Gate Sweep Window" if it exists
            if gate_sweep_window_label:
                gate_sweep_window_label.destroy()
                gate_sweep_window_label = None
            if gate_sweep_window_text:
                gate_sweep_window_text.destroy()
                gate_sweep_window_text = None

            # Remove gate sweep window if there
            if line3:
                line3.remove()
                line3 = None
                canvas.draw()

        case 'Waveform Recovery':
            output_mode = 2
            ax1.set_ylabel('Pulse Input Amplitude (Volts)', color=color_ch1)
            mcc.set_control(4, output_mode)
            auto_button.configure(state='normal')

            # Add "Gate Sweep Window" if it doesn't exist
            if not gate_sweep_window_label:
                gate_sweep_window_label = tkinter.Label(root, text="Gate Sweep Window")
                gate_sweep_window_label.grid(row=6, column=matplot_columnspan+1, sticky=left_column_sticky)

            if not gate_sweep_window_text:
                gate_sweep_window_text = tkinter.Entry(root, width=8)
                gate_sweep_window_text.insert(tkinter.END, str(INITIAL_GATEWIDTH * 2))
                gate_sweep_window_text.grid(row=6, column=matplot_columnspan+2, sticky=right_column_sticky)
                gate_sweep_window_text.bind('<Return>', update_green_pulse)
            
            update_green_pulse(None)

#########################################################################
## Quit the application
def quit(tk_root, instrument):
    exit_event.set()
    tk_root.destroy()
    instrument.relinquish_ownership()

#########################################################################
def update_plot():
    global line1, line2, line3, line4

    while True:

        if exit_event.is_set():
            break
        try:
            if mode.get() == "Average Output" and bc_mode.get() == "Waveform Recovery":
                # Remove Moku Channel 1 and 2 input waveforms
                if line1:
                    line1.remove()
                    line1 = None
                if line2:
                    line2.remove()
                    line2 = None
                if line3:
                    line3.remove()
                    line3 = None
                average_waveform()
            else:
                # Remove recovery waveform, if there, and re-initialize line1 and line2 for Moku channels 1 and 2 
                if line4:
                    line4.remove()
                    line4 = None
                    color_ch1 = 'tab:blue'
                    line1, = ax1.plot([],[],color=color_ch1)
                    color_ch2 = 'tab:red'
                    line2, = ax2.plot([],[],color=color_ch2)

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

                if bc_mode.get() == "Waveform Recovery":
                    update_green_pulse(None)
                elif bc_mode.get() == "Static Gate" and mode.get() == "Average Output":
                    gate_width = float(gate_width_text.get()) 
                    gate_width_bits = math.ceil(gate_width*ns/period)
                    
                    gain = int(float(gain_text.get())*2**16)/2**16
                    
                    out_text.delete("1.0", "end")
                    warning_text.delete("1.0", "end")
                    if abs(data['ch1'][0]) > range_hardware/2*saturation_threshold:
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
def update_green_pulse(event):
    global line3

    # Get the new gate sweep width from the entry box
    try:
        gate_sweep_width = float(gate_sweep_window_text.get())  # Value in ns
    except ValueError:
        warning_text.insert("1.0", "Invalid Gate Sweep Width Value")
        warning_text.insert("1.0", "Invalid Gate Sweep Width Value")
        return

    # Update the green pulse based on the new width
    data = osc.get_data()
    data_time_ns = np.array(data['time']) / ns
    gate_signal = np.array(data['ch2'])
    green_pulse = np.zeros_like(gate_signal)

    # Identify all pulses in the gate_signal
    pulse_indices = np.where(gate_signal > 0)[0]
    if len(pulse_indices) == 0:
        return  # No pulses detected

    pulse_start_indices = [pulse_indices[0]]
    for i in range(1, len(pulse_indices)):
        if pulse_indices[i] != pulse_indices[i - 1] + 1:
            pulse_start_indices.append(pulse_indices[i])

    # Create green pulses starting at the same indices as the gate_signal pulses
    for start_idx in pulse_start_indices:
        pulse_length_indices = int(gate_sweep_width / (data_time_ns[1] - data_time_ns[0]))  # Convert ns to indices
        new_end_idx = min(len(green_pulse), start_idx + pulse_length_indices)

        # Set the green pulse
        green_pulse[start_idx:new_end_idx] = max(gate_signal)

        # Remove the existing green pulse
        if line3:
            line3.set_data(data_time_ns, green_pulse)
        else:
            line3, = ax2.plot(data['time'], green_pulse, color='green', linestyle='--', label="Green Pulse (Dynamic Width)")

        line3.figure.canvas.flush_events()

#########################################################################
def average_waveform():
    global line4

    trg_delay = float(trg_delay_text.get())
    trg_delay_bits = math.ceil(trg_delay*ns/period)
    gate_width = float(gate_width_text.get()) 
    gate_width_bits = math.ceil(gate_width*ns/period)
    gate_sweep_width = float(gate_sweep_window_text.get())
    gate_sweep_width_bits = math.ceil(gate_sweep_width*ns/period)
    num_gates = int(float(avg_length_text.get()))
    gain = float(gain_text.get())
    scale_factor = 1/((1/gain)*(1/num_gates)*(math.ceil(1/gate_width_bits))) # Calcuate scale factor to bring output value back to original scale

    # Ensure line4 is initialized
    if line4 is None:
        line4, = ax1.plot([], [], color="blue", marker="o", label="Averaged Data")
        ax1.set_xlabel("Time [ns]")
        ax1.set_ylabel("Average Output Amplitude [Volts]", color="blue")

    i_values = []
    ave_values = []

    for i in range(trg_delay_bits, (trg_delay_bits+gate_sweep_width_bits)-gate_width_bits, gate_width_bits):
        if mode.get() != "Average Output":
            break
        mcc.set_control(1,i) # Set trigger delay as it is increased throughout the pulse duration each loop
        data = osc.get_data() # Collect oscilloscope data
        ave = sum(data['ch1'])/len(data['ch1']) # Calculate the average of output data points to get a single output value
        if abs(ave) > range_hardware/2*saturation_threshold:
            warning_text.insert(tkinter.END, 'Possible saturation detected, please reduce gain' )
        ave = ave/scale_factor # Implement scale factor to return output signal to original magnitude

        i_values.append(i*period/ns) # Store trigger delay value for x axis
        ave_values.append(ave) # Store output average values
        i_values = [x-i_values[0] for x in i_values] # Normalize trigger delay so that it starts at 0

        # Update the plot dynamically
        line4.set_data(i_values, ave_values)
        ax1.set_xlim([0, max(i_values) + 10])  # Adjust X-axis dynamically
        ax1.set_ylim([min(ave_values)*0.9, max(ave_values)*1.1])  # Adjust Y-axis
        canvas.draw()  # Redraw the canvas

        time.sleep(1)

    #mean_ave_values = sum(ave_values)/len(ave_values)
    #print('Average value = ', mean_ave_values)
    #print('Standard Deviation = ', np.std(ave_values))
    
#########################################################################

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
tkinter.Label(root, text="Trigger Level (Volts)").grid(row=3, column=matplot_columnspan+1, sticky=left_column_sticky) 
trg_level_text = tkinter.Entry(root, width = 8)
trg_level_text.insert(tkinter.END, str(INITIAL_TRIGGER_LEVEL))
trg_level_text.grid(row=3, column=matplot_columnspan+2, sticky = right_column_sticky) 
trg_level_text.bind('<Return>',update_trg_level)

## Boxcar Gate Width input boxes
tkinter.Label(root, text="Gate Width (ns)").grid(row=5, column=matplot_columnspan+1, sticky=left_column_sticky) 
gate_width_text = tkinter.Entry(root, width = 8)
gate_width_text.insert(tkinter.END, str(INITIAL_GATEWIDTH))
gate_width_text.grid(row=5, column=matplot_columnspan+2, sticky=right_column_sticky) 
gate_width_text.bind('<Return>',update_gate_width)

## Boxcar Average Length input boxes
tkinter.Label(root, text="# of Avg.").grid(row=4, column=matplot_columnspan+1, sticky=left_column_sticky) 
avg_length_text = tkinter.Entry(root, width = 8)
avg_length_text.insert(tkinter.END, str(INITIAL_AVERAGE_LENGTH))
avg_length_text.grid(row=4, column=matplot_columnspan+2, sticky=right_column_sticky)
avg_length_text.bind('<Return>',update_avg_length)

## Gain input box
tkinter.Label(root, text="Output Gain").grid(row=11, column=matplot_columnspan+1, sticky=left_column_sticky) 
gain_text = tkinter.Entry(root, width = 8)
gain_text.insert(tkinter.END, str(INITIAL_OUTPUT_GAIN))
gain_text.grid(row=11, column=matplot_columnspan+2, sticky=right_column_sticky) 
gain_text.bind('<Return>',update_gain)

## Trigger Delay input boxes
tkinter.Label(root, text="Pulse Trg Delay [ns]").grid(row=7, column=matplot_columnspan+1, sticky=left_column_sticky) 
trg_delay_text = tkinter.Entry(root, width = 8)
trg_delay_text.insert(tkinter.END, str(INITIAL_TRIGGER_DELAY))
trg_delay_text.grid(row=7, column=matplot_columnspan+2, sticky=right_column_sticky)
trg_delay_text.bind('<Return>',update_trg_delay)

## Trigger Delay Auto button
auto_button = tkinter.Button(root, text = "Auto",  command = auto_delay, state = 'normal')
auto_button.grid(row=7, column=matplot_columnspan+3, sticky=right_column_sticky)

## Boxcar mode select
tkinter.Label(root, text="Static/WR").grid(row=2, column=matplot_columnspan+1, sticky=left_column_sticky)
bc_mode_options = ["Static Gate", "Waveform Recovery"]
bc_mode = ttk.Combobox(root, values=bc_mode_options, height = len(bc_mode_options), width = 18)
bc_mode.bind("<<ComboboxSelected>>", update_boxcar_mode)
bc_mode.set("Static Gate")
bc_mode.grid(row=2, column=matplot_columnspan+2, sticky=right_column_sticky) 

## Quit button
tkinter.Button(root, text = "Quit", command = lambda: quit(root, m)).grid(row=matplot_rowspan + 2, column=matplot_columnspan+3, sticky='e')

#################################################################

## Read out
out_text = tkinter.Text(root, height = 1, width = 14)
out_text.grid(row=9, column=matplot_columnspan+2, sticky=right_column_sticky)
out_text.config(font="TkTextFont 14 normal")
tkinter.Label(root, text="Averaged Output [mV]").grid(row=9, column=matplot_columnspan+1, sticky=left_column_sticky) 

## Warning text box
warning_text = tkinter.Text(root, height = 5, width = 20, font="TkTextFont 14 normal")
warning_text.grid(row=matplot_rowspan-1, column=matplot_columnspan+1, columnspan = 3, sticky = tkinter.W+tkinter.E)

####################################################################

print('Initializing...')

mcc.set_control(0,int(INITIAL_TRIGGER_LEVEL/resolution) )
mcc.set_control(1,int(INITIAL_TRIGGER_DELAY*ns/period) )
mcc.set_control(2,int(INITIAL_GATEWIDTH*ns/period) )
mcc.set_control(3,int(INITIAL_AVERAGE_LENGTH) )
mcc.set_control(4,INITIAL_OUTPUT_MODE) 
mcc.set_control(5,int(INITIAL_OUTPUT_GAIN*2**16) )

ax1.set_ylabel('Pulse Input Amplitude (Volts)', color=color_ch1)
    
#########################################################################

## Matplotlib box
canvas.get_tk_widget().grid(row=0, column=0, rowspan=60, columnspan=120, ipadx=100, ipady=20)
tkinter.mainloop()