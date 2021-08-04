---
title: generate_waveform | Waveform Generator
additional_doc: null
description: Configures the output waveform.
method: post
name: generate_waveform
parameters:
- default: null
  description: Target channel
  name: channel
  param_range:
   mokugo: 1, 2
   mokupro: 1, 2, 3, 4
  type: integer
  unit: null
- default: null
  description: Waveform type
  name: type
  param_range: Off, Sine, Square, Ramp, Pulse, DC
  type: string
  unit: null
- default: 1
  description: Waveform peak-to-peak amplitude
  name: amplitude
  param_range: 4e-3 to 10
  type: number
  unit: V
- default: 10000
  description: Waveform frequency
  name: frequency
  param_range: 1e-3 to 20e6
  type: number
  unit: Hz
- default: 0
  description: DC offset applied to the waveform
  name: offset
  param_range: -5 to 5
  type: number
  unit: V
- default: 0
  description: Waveform phase offset
  name: phase
  param_range: 0 to 360
  type: number
  unit: Deg
- default: 50
  description: Duty cycle as percentage (Only for Square wave)
  name: duty
  param_range: 0.0 to 100.0
  type: number
  unit: '%'
- default: 50.0
  description: Fraction of the cycle rising
  name: symmetry
  param_range: 0.0 to 100.0
  type: number
  unit: '%'
- default: 0
  description: DC Level. (Only for DC waveform)
  name: dc_level
  param_range: null
  type: number
  unit: null
- default: 0
  description: Edge-time of the waveform (Only for Pulse wave)
  name: edge_time
  param_range: 16e-9 to pulse width
  type: number
  unit: null
- default: 0
  description: Pulse width of the waveform (Only for Pulse wave)
  name: pulse_width
  param_range: null
  type: number
  unit: null
- default: true
  description: Disable all implicit conversions and coercions.
  name: strict
  param_range: null
  type: boolean
  unit: null
summary: generate_waveform
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