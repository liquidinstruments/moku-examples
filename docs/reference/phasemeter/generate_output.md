---
title: generate_output
additional_doc: null
description: Generate a signal on the specified output channel
method: post
name: generate_output
parameters:
- default: null
  description: Target channel
  name: channel
  param_range: 1, 2, 3, 4
  type: integer
  unit: null
- default: 1
  description: Waveform peak-to-peak amplitude
  name: amplitude
  param_range: 1e-3 to 10
  type: number
  unit: V
- default: 10000
  description: Waveform frequency
  name: frequency
  param_range: 1e-3 to 100e6
  type: number
  unit: Hz
- default: 0
  description: Phase offset of the wave
  name: phase
  param_range: 0 to 360
  type: number
  unit: Deg
- default: true
  description: Locks the phase of the generated sinewave to the measured phase of the input signal
  name: phase_locked
  param_range: null
  type: boolean
  unit: null
- default: Sine
  description: Type of output signal
  name: signal
  param_range: Sine, Phase
  type: string
  unit: null
- default: 0.001
  description: Phase scaling (Only used when the output signal is set to Phase)
  name: phase_scaling
  param_range: 10e-9 to 100e3
  type: number
  unit: V/cyc
- default: true
  description: Disable all implicit conversions and coercions.
  name: strict
  param_range: null
  type: boolean
  unit: null
summary: generate_output 
available_on: "mokupro"
---




<headers/>
<parameters/>

Examples,

<code-group>
<code-block title="Python">
```python{4-7}
from moku.instruments import WaveformGenerator

i = WaveformGenerator('192.168.###.###', force_connect=False)
# Generate a sine wave on channel 1, 0.5 Vpp, 5 kHz
# Generate a square wave on channel 2, 1 Vpp, 1 kHz, 50% duty cycle
i.generate_waveform(channel=1, type='Sine', amplitude=0.5, frequency=5e3)
i.generate_waveform(channel=2, type='Square', amplitude=1.0, frequency=1e3, duty=50)
```
</code-block>

<code-block title="MATLAB">
```matlab{2-7}
m = MokuWaveformGenerator('192.168.###.###', false);
% Generate a sine wave on Channel 1
% 1Vpp, 10kHz, 0V offset
m.generate_waveform(1, 'Sine','amplitude', 1, 'frequency',1000,'offset',0.2);
% Generate a sine wave on Channel 2
% 1Vpp, 10kHz, 0V offset, 50% duty cycle
m.generate_waveform(2, 'Sine', 'amplitude',1,'frequency', 10e3);
```
</code-block>
</code-group>