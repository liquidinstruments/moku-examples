---
title: set_modulation | Waveform Generator
additional_doc: null
description: Set up modulation on an output channel.
method: post
name: set_modulation
parameters:

- default: null
  description: Target channel
  name: channel
  param_range:
   mokugo: 1, 2
   mokulab: 1, 2
   mokupro: 1, 2, 3, 4
  type: integer
  unit: null
- default: null
  description: Modulation type
  name: type
  param_range: Amplitude, Frequency, Phase, PulseWidth
  type: string
  unit: null
- default: null
  description: Modulation source
  name: source
  param_range: 
    mokugo: Input1, Input2, Output1, Output2, Internal
    mokulab: Input1, Input2, Output1, Output2, Internal
    mokupro: Input1, Input2, Input3, Input4, Output1, Output2, Output3, Output4, Internal
  type: string
  unit: null
- default: 0
  description: 'Modulation depth (depends on modulation type): Percentage modulation
    depth, Frequency Deviation/Volt or +/- phase shift/Volt'
  name: depth
  param_range: null
  type: number
  unit: null
- default: 10000000
  description: Frequency of internally-generated sine wave modulation. This parameter
    is is only used when the modulation source is set to internal.
  name: frequency
  param_range:
   mokugo: 0 to 5e6
   mokulab: 0 to 62.5e6
   mokupro: 0 t0 125e6
  type: number
  unit: Hz
- default: true
  description: Disable all implicit conversions and coercions.
  name: strict
  param_range: null
  type: boolean
  unit: null
summary: set_modulation
---




<headers/>
<parameters/>


<code-group>
<code-block title="Python">
```python
from moku.instruments import WaveformGenerator

i = WaveformGenerator('192.168.###.###')
i.generate_waveform(channel=1, type='Sine', amplitude=0.5, frequency=5e3)
i.generate_waveform(channel=2, type='Sine', amplitude=1.0, frequency=1e6)
# Configure amplitude modulation on channel 1. 
# Use internal reference as modulation source, modulation depth 50%, 
# modulated at a frequency of 1Hz
i.set_modulation(channel=1, type='Amplitude', source='Internal', depth=50,
                     frequency=1)
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuWaveformGenerator('192.168.###.###');
% Generate a sine wave on Channel 1
% 1Vpp, 10kHz, 0V offset
m.generate_waveform(1, 'Sine','amplitude', 1, 'frequency',1000,'offset',0.2);
% Generate a sine wave on Channel 2
% 1Vpp, 10kHz, 0V offset, 50% duty cycle
m.generate_waveform(2, 'Sine', 'amplitude',1,'frequency', 10e3);
% Amplitude modulate the Channel 1 Sinewave with another internally-
% generated sinewave. 50% modulation depth at 1Hz.
m.set_modulation(2, 'Frequency', 'Internal', 'depth',1e3, 'frequency', 1);
```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"channel": 1, "type": "Amplitude", "source": "Internal", "frequency": 1, "depth": 50}'\
        http://<ip>/api/waveformgenerator/set_modulation
```
</code-block>

</code-group>

### Sample response 
```json
{
  "depth": 50,
  "frequency": 1.0,
  "source": "Internal",
  "type": "Amplitude"
}
```