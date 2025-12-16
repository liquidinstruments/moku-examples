---
pageClass: wide-page
sidebarDepth: 1
---
# Examples for Python

## Arbitrary Waveform Generator

### arbitrarywavegen_basic.py

This example demonstrates how you can generate and output arbitrary waveforms using Moku AWG

<<< @/docs/api/moku-examples/python-api/arbitrarywavegen_basic.py

## Custom Instrument

### custom_instrument_adder.py

This example demonstrates how you can configure Custom Instrument, using Multi-Instrument mode to run the
[Adder example](https://github.com/liquidinstruments/moku-examples/tree/main/mc/Basic/Adder)

<<< @/docs/api/moku-examples/python-api/custom_instrument_adder.py

### custom_instrument_arithmetic.py

This example demonstrates how you can configure Custom Instrument, using Multi-Instrument mode to run the
[Arithmetic Unit example](https://github.com/liquidinstruments/moku-examples/tree/main/mc/Moderate/ArithmeticUnit)

<<< @/docs/api/moku-examples/python-api/custom_instrument_arithmetic.py

## Data Logger

### datalogger_basic.py

This example demonstrates use of the Datalogger instrument to log time-series
voltage data to a file.

<<< @/docs/api/moku-examples/python-api/datalogger_basic.py

## Data Streaming

### datalogger_streaming.py

This example demonstrates use of the Datalogger instrument to stream time-series
voltage data.

<<< @/docs/api/moku-examples/python-api/datalogger_streaming.py

### lock_in_amplifier_streaming.py

This example demonstrates use of the Lockin Amplifier instrument to demodulate Input signal with a
reference signal from local oscillator and stream the generated waveform from auxiliary output

<<< @/docs/api/moku-examples/python-api/lock_in_amplifier_streaming.py

### mim_dl_lia_streaming.py

This example demonstrates data streaming feature in Multi-instrument mode, deploying
Datalogger in slot1 and Lockin Amplifier in slot2.

<<< @/docs/api/moku-examples/python-api/mim_dl_lia_streaming.py

## Digital Filter Box

### digital_filter_box_plotting.py

This example demonstrates how you can configure the Digital Filter Box instrument
to filter and display two signals.

<<< @/docs/api/moku-examples/python-api/digital_filter_box_plotting.py

## Frequency Response Analyzer

### freq_response_analyzer_basic.py

This example demonstrates how you can generate output sweeps using the
Frequency Response Analyzer instrument, and view one frame of the transfer
function data.

<<< @/docs/api/moku-examples/python-api/freq_response_analyzer_basic.py

### freq_response_analyzer_plotting.py

This example demonstrates how you can generate output sweeps using the
Frequency Response Analyzer instrument, and view transfer function data
in real-time.

<<< @/docs/api/moku-examples/python-api/freq_response_analyzer_plotting.py

## Gigabit Streamer

### gigabit_streamer_plotting.py

This example demonstrates how you can stream on a Moku in loopback,
receiving and transmitting data using Moku Gigabit Streamer, using
the Oscilloscope to generate and view the data in real-time.

<<< @/docs/api/moku-examples/python-api/gigabit_streamer_plotting.py

## Gigabit Streamer+

### gigabit_streamer_plus_basic.py

This example demonstrates how to transmit and receive data using
Moku Gigabit Streamer+ in single instrument mode.

<<< @/docs/api/moku-examples/python-api/gigabit_streamer_plus_basic.py

## Laser Lock Box

### laser_lock_box_basic.py

This example demonstrates how you can configure the Laser Lock Box
Instrument and monitor the signals at Input 1 and Input 2.

<<< @/docs/api/moku-examples/python-api/laser_lock_box_basic.py

### laser_lock_box_plotting.py

This example demonstrates how you can configure the Laser Lock Box
Instrument and monitor the signals at Input 1 and Input 2.

<<< @/docs/api/moku-examples/python-api/laser_lock_box_plotting.py

## Lock-in Amplifier

### lock_in_amplifier_basic.py

This example demonstrates how you can configure the Lock-in Amplifier
instrument to demodulate an input signal from Input 1 with the reference
signal from the Local Oscillator to extract the X component and generate
a sine wave on the auxiliary output

<<< @/docs/api/moku-examples/python-api/lock_in_amplifier_basic.py

### lock_in_amplifier_plotting.py

This example demonstrates how you can configure the Lock-in Amplifier
instrument to demodulate an input signal from Input 1 with the reference
signal from the Local Oscillator to extract the X component and generate
a sine wave on the auxiliary output

<<< @/docs/api/moku-examples/python-api/lock_in_amplifier_plotting.py

## Logic Analyzer

### logic_analyzer_plotting.py

This example demonstrates how you can configure the Logic
Analyzer instrument to retrieve a single frame of data for
all 16 channels

<<< @/docs/api/moku-examples/python-api/logic_analyzer_plotting.py

## Multi-instrument Mode

### mim_wg_osc.py

Multi-instrument Mode on Moku:Go (two-slot), deploying the Waveform Generator
and Oscilloscope at once. This example is easily ported to Moku:Pro by changing
the "platform_id" to one supported by that hardware.

<<< @/docs/api/moku-examples/python-api/mim_wg_osc.py

### mim_wg_sa.py

Multi-instrument Mode on Moku:Go (two slot), deploying the Waveform Generator
and Spectrum Analyzer at once. This example is easily ported to Moku:Pro by changing
the "platform_id" to one supported by that hardware.

<<< @/docs/api/moku-examples/python-api/mim_wg_sa.py

## Neural Network

### neuralnetwork_simplesine.py

This script demonstrates how to use the Neural Network instrument to generate a ramp
wave and process it through the uploaded neural network, finally viewing the output
in the oscilloscope. This uses the network generated in the
[Simple Sine wave example](https://apis.liquidinstruments.com/mnn/examples/Simple_sine.html)

<<< @/docs/api/moku-examples/python-api/neuralnetwork_simplesine.py

## Oscilloscope

### oscilloscope_basic.py

This script demonstrates how to use the Oscilloscope instrument
to retrieve a single frame of dual-channel voltage data.

<<< @/docs/api/moku-examples/python-api/oscilloscope_basic.py

### oscilloscope_plotting.py

This example demonstrates how you can configure the Oscilloscope instrument,
and view triggered time-voltage data frames in real-time.

<<< @/docs/api/moku-examples/python-api/oscilloscope_plotting.py

### oscilloscope_deep_memory_mode.py

This example demonstrates how to acquire data using the deep memory mode in the
Moku Oscilloscope.

<<< @/docs/api/moku-examples/python-api/oscilloscope_deep_memory_mode.py

## Phasemeter

### phasemeter_basic.py

This example demonstrates how you can configure the Phasemeter
instrument to measure 4 independent signals.

<<< @/docs/api/moku-examples/python-api/phasemeter_basic.py

### phasemeter_plotting.py

This example demonstrates how you can configure the Phasemeter
instrument to measure 2 independent signals and display the results in real-time.

<<< @/docs/api/moku-examples/python-api/phasemeter_plotting.py

### phasemeter_streaming.py

This example demonstrates how you can configure the Phasemeter
instrument to measure a signal, stream and display the results.

<<< @/docs/api/moku-examples/python-api/phasemeter_streaming.py

## PID Controller

### pidcontroller_basic.py

This script demonstrates how to configure one of the two PID Controllers
in the PID Controller instrument. Configuration is done by specifying
frequency response characteristics of the controller.

<<< @/docs/api/moku-examples/python-api/pidcontroller_basic.py

### pidcontroller_plotting.py

This script demonstrates how to configure both PID Controllers
in the PID Controller instrument. Configuration on the Channel 1
PID is done by specifying frequency response characteristics,
while Channel 2 specifies the gain characteristics.

The output response of each PID Controller channel is plotted
in real-time.

<<< @/docs/api/moku-examples/python-api/pidcontroller_plotting.py

## Programmable Power Supplies

### powersupply_basic.py

This example will demonstrate how to configure the power supply
units of the Moku:Go.

<<< @/docs/api/moku-examples/python-api/power_supply_basic.py

## Spectrum Analyzer

### spectrumanalyzer_basic.py

This example demonstrates how you can use the Spectrum Analyzer instrument to
to retrieve a single spectrum data frame over a set frequency span.

<<< @/docs/api/moku-examples/python-api/spectrumanalyzer_basic.py

### spectrumanalyzer_plotting.py

This example demonstrates how you can configure the Spectrum Analyzer
instrument and plot its spectrum data in real-time.

<<< @/docs/api/moku-examples/python-api/spectrumanalyzer_plotting.py

## Time and Frequency Analyzer

### timefrequencyanalyzer_basic.py

This example demonstrates how you can use the Time and Frequency Analyzer instrument to
to retrieve and view the statistics of the intervals.

<<< @/docs/api/moku-examples/python-api/timefrequencyanalyzer_basic.py

### timefrequencyanalyzer_plotting.py

This example demonstrates how you can configure the Time and Frequency Analyzer
instrument and plot histogram data frames in real-time.

<<< @/docs/api/moku-examples/python-api/timefrequencyanalyzer_plotting.py

## Waveform Generator

### waveformgenerator_basic.py

This example demonstrates how you can use the Waveform Generator
instrument to generate a sinewave on Channel 1 and a square wave on Channel 2.

<<< @/docs/api/moku-examples/python-api/waveformgenerator_basic.py

### waveformgenerator_modulation.py

This example demonstrates how you can use the Waveform Generator instrument
to generate an amplitude modulated sine wave on Channel 1, and a sweep
modulated sine wave on Channel 2.

<<< @/docs/api/moku-examples/python-api/waveformgenerator_modulation.py

### waveformgenerator_triggered.py

This example demonstrates how you can use the Waveform Generator instrument
to generate a gated sine wave on Channel 1, and a swept frequency square wave
on Channel 2.

<<< @/docs/api/moku-examples/python-api/waveformgenerator_triggered.py
