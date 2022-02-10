---
title: generate_output
additional_doc: The available outputs are either a sine wave, optionally locked to the incoming
  signal; or a voltage proportional to the current phase measurement value
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
  unit: Vpp
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
  description: (Upcoming) Type of output signal
  name: signal
  param_range: Sine, Phase
  type: string
  unit: null
- default: 0.001
  description: (Upcoming) Phase scaling (only used when the output signal is set to Phase)
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

### Examples


<code-group>
<code-block title="Python">
```python{4-7}
from moku.instruments import MokuPhasemeter

i = MokuPhasemeter('192.168.###.###', force_connect=False)
# Generate a sine wave on channel 1, 0.5 Vpp, 5 kHz
# Generate a sine wave on channel 2, 1 Vpp, 1 MHz
i.generate_output(channel=1, amplitude=0.5, frequency=5e3)
i.generate_output(channel=2,  amplitude=1.0, frequency=1e6)

i.set_frontend(channel=1, impedance='50Ohm', coupling='DC', range='4Vpp')
```
</code-block>

<code-block title="MATLAB">
```matlab{2-8}
i = MokuPhasemeter('192.168.###.###', false);
% Generate a sine wave on Channel 1
% 0.5 Vpp, 10 kHz
i.generate_output(1, 0.5, 10e3);

% Generate a sine wave on Channel 2
% 1 Vpp, 1 MHz
i.generate_output(2, 1, 1e6);

i.set_frontend(1,'50Ohm','DC','4Vpp');
```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"channel": 1, "amplitude": 1, "frequency": 20e3}'\
        http://<ip>/api/phasemeter/generate_output
```
</code-block>

</code-group>