---
pageClass: wide-page
sidebarDepth: 1
---

# Examples for Python

## Arbitrary Waveform Generator

### arbitrarywavegen_basic.py

This example demonstrates how you can generate and output arbitrary waveforms using Moku AWG

<<< @/docs/api/examples/python/arbitrarywavegen_basic.py

## Cloud Compile

### cloud_compile_adder.py

This example demonstrates how you can configure Cloud Compile, using Multi-Instrument mode to run the
[Adder example](https://gitlab.com/liquidinstruments/cloud-compile/examples/-/tree/main/adder)

<<< @/docs/api/examples/python/cloud_compile_adder.py

### cloud_compile_arithmetic.py

This example demonstrates how you can configure Cloud Compile, using Multi-Instrument mode to run the
[Arithmetic Unit example](https://gitlab.com/liquidinstruments/cloud-compile/examples/-/tree/main/arithmetic_unit)

<<< @/docs/api/examples/python/cloud_compile_arithmetic.py

## Data Logger

### datalogger_basic.py

This example demonstrates use of the Datalogger instrument to log time-series
voltage data to a file.

<<< @/docs/api/examples/python/datalogger_basic.py

## Data Streaming

### datalogger_streaming.py

This example demonstrates use of the Datalogger instrument to stream time-series
voltage data.

<<< @/docs/api/examples/python/datalogger_streaming.py

### lock_in_amplifier_streaming.py

This example demonstrates use of the Lockin Amplifier instrument to demodulate Input signal with a
reference signal from local oscillator and stream the generated waveform from auxiliary output

<<< @/docs/api/examples/python/lock_in_amplifier_streaming.py

### mim_dl_lia_streaming.py

This example demonstrates data streaming feature in Multi-instrument mode, deploying
Datalogger in slot1 and Lockin Amplifier in slot2.

<<< @/docs/api/examples/python/mim_dl_lia_streaming.py

### phasemeter_streaming.py

This example demonstrates how you can configure the Phasemeter
instrument to measure a signal, stream and display the results.

<<< @/docs/examples/python/phasemeter_streaming.py

## Digital Filter Box
### digital_filter_box_plotting.py

This example demonstrates how you can configure the Digital Filter Box instrument
to filter and display two signals.

<<< @/docs/api/examples/python/digital_filter_box_plotting.py

## FIR Filter
### fir_filter_builder_basic.py

This example demonstrates how you can configure the FIR Filter instrument 
to filter two signals.

<<< @/docs/examples/python/fir_filter_builder_basic.py

### fir_filter_builder_plotting.py

This example demonstrates how you can configure the FIR Filter instrument 
to filter and display two signals.

<<< @/docs/examples/python/fir_filter_builder_plotting.py

## Frequency Response Analyzer

### freq_response_analyzer_basic.py

This example demonstrates how you can generate output sweeps using the
Frequency Response Analyzer instrument, and view one frame of the transfer
function data.

<<< @/docs/api/examples/python/freq_response_analyzer_basic.py

### freq_response_analyzer_plotting.py

This example demonstrates how you can generate output sweeps using the
Frequency Response Analyzer instrument, and view transfer function data
in real-time.

<<< @/docs/api/examples/python/freq_response_analyzer_plotting.py

## Laser Lock Box

### laser_lock_box_basic.py

This example demonstrates how you can configure the Laser Lock Box
Instrument and monitor the signals at Input 1 and Input 2.

<<< @/docs/api/examples/python/laser_lock_box_basic.py

### laser_lock_box_plotting.py

This example demonstrates how you can configure the Laser Lock Box
Instrument and monitor the signals at Input 1 and Input 2.

<<< @/docs/api/examples/python/laser_lock_box_plotting.py

## Lock-in Amplifier

### lock_in_amplifier_basic.py

This example demonstrates how you can configure the Lock-in Amplifier
instrument to demodulate an input signal from Input 1 with the reference
signal from the Local Oscillator to extract the X component and generate
a sine wave on the auxiliary output

<<< @/docs/api/examples/python/lock_in_amplifier_basic.py

### lock_in_amplifier_plotting.py

This example demonstrates how you can configure the Lock-in Amplifier
instrument to demodulate an input signal from Input 1 with the reference
signal from the Local Oscillator to extract the X component and generate
a sine wave on the auxiliary output

<<< @/docs/api/examples/python/lock_in_amplifier_plotting.py

## Logic Analyzer

### logic_analyzer_plotting.py

This example demonstrates how you can configure the Logic
Analyzer instrument to retrieve a single frame of data for
all 16 channels

<<< @/docs/api/examples/python/logic_analyzer_plotting.py

## Multi-instrument Mode

### mim_wg_osc.py

Multi-instrument Mode on Moku:Go (two-slot), deploying the Waveform Generator
and Oscilloscope at once. This example is easily ported to Moku:Pro by changing
the "platform_id" to one supported by that hardware.

<<< @/docs/api/examples/python/mim_wg_osc.py

### mim_wg_sa.py

Multi-instrument Mode on Moku:Go (two slot), deploying the Waveform Generator
and Spectrum Analyzer at once. This example is easily ported to Moku:Pro by changing
the "platform_id" to one supported by that hardware.

<<< @/docs/api/examples/python/mim_wg_sa.py

## Oscilloscope

### oscilloscope_basic.py

This script demonstrates how to use the Oscilloscope instrument
to retrieve a single frame of dual-channel voltage data.

<<< @/docs/api/examples/python/oscilloscope_basic.py

### oscilloscope_plotting.py

This example demonstrates how you can configure the Oscilloscope instrument,
and view triggered time-voltage data frames in real-time.

<<< @/docs/api/examples/python/oscilloscope_plotting.py

## Phasemeter

### phasemeter_basic.py

This example demonstrates how you can configure the Phasemeter
instrument to measure 4 independent signals.

<<< @/docs/api/examples/python/phasemeter_basic.py

## PID Controller

### pidcontroller_basic.py

This script demonstrates how to configure one of the two PID Controllers
in the PID Controller instrument. Configuration is done by specifying
frequency response characteristics of the controller.

<<< @/docs/api/examples/python/pidcontroller_basic.py

### pidcontroller_plotting.py

This script demonstrates how to configure both PID Controllers
in the PID Controller instrument. Configuration on the Channel 1
PID is done by specifying frequency response characteristics,
while Channel 2 specifies the gain characteristics.

The output response of each PID Controller channel is plotted
in real-time.

<<< @/docs/api/examples/python/pidcontroller_plotting.py

## Programmable Power Supplies

### powersupply_basic.py

This example will demonstrate how to configure the power supply
units of the Moku:Go.

<<< @/docs/api/examples/python/power_supply_basic.py

## Spectrum Analyzer

### spectrumanalyzer_basic.py

This example demonstrates how you can use the Spectrum Analyzer instrument to
to retrieve a single spectrum data frame over a set frequency span.

<<< @/docs/api/examples/python/spectrumanalyzer_basic.py

### spectrumanalyzer_plotting.py

This example demonstrates how you can configure the Spectrum Analyzer
instrument and plot its spectrum data in real-time.

<<< @/docs/api/examples/python/spectrumanalyzer_plotting.py

## Waveform Generator

### waveformgenerator_basic.py

This example demonstrates how you can use the Waveform Generator
instrument to generate a sinewave on Channel 1 and a square wave on Channel 2.

<<< @/docs/api/examples/python/waveformgenerator_basic.py

### waveformgenerator_modulation.py

This example demonstrates how you can use the Waveform Generator instrument
to generate an amplitude modulated sine wave on Channel 1, and a sweep
modulated sine wave on Channel 2.

<<< @/docs/api/examples/python/waveformgenerator_modulation.py
