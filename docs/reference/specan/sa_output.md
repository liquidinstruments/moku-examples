---
additional_doc: null
description: Generate a waveform on the output channels.
method: post
name: sa_output
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
  description: Waveform peak-to-peak amplitude
  name: amplitude
  param_range: null
  type: number
  unit: null
- default: null
  description: Frequency of the wave
  name: frequency
  param_range: 0 to 30e6
  type: number
  unit: Hz
- default: true
  description: Disable all implicit conversions and coercions.
  name: strict
  param_range: null
  type: boolean
  unit: null
summary: sa_output
---

<headers/>
<parameters/>

Usage in clients, 

<code-group>
<code-block title="Python">
```python
from moku.instruments import SpectrumAnalyzer
i = SpectrumAnalyzer('192.168.###.###', force_connect=False)
# Generate a Sine wave on output channel 1 
i.sa_output(channel=1, amplitude=0.5, frequency=1e5)
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuSpectrumAnalyzer('192.168.###.###', true);
% Generate a Sine wave on output channel 1 
m.sa_output(1, 0.5, 1e5)
```
</code-block>
</code-group>