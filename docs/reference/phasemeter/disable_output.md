---
title: disable_output
additional_doc: null
description: Turn off the output channels
method: post
name: disable_modulation
parameters:
- default: null
  description: Target channel
  name: channel
  param_range: 1, 2, 3, 4
  type: integer
  unit: null
- default: true
  description: Disable all implicit conversions and coercions.
  name: strict
  param_range: null
  type: boolean
  unit: null
summary: disable_output
available_on: "mokupro"
---



<headers/>
<parameters/>

Examples,

<code-group>
<code-block title="Python">
```python{5,6}
from moku.instruments import MokuPhasemeter

i = MokuPhasemeter('192.168.###.###', force_connect=False)
i.generate_waveform(channel=1, type='Sine', amplitude=0.5, frequency=5e3)
# disable modulation on output channel 1
i.disable_modulation(channel=1)
```
</code-block>

<code-block title="MATLAB">
```matlab{3,4}
m = MokuPhasemeter('192.168.###.###', false);
m.generate_waveform(1, 'Sine', 'amplitude', 1, 'frequency', 1e3, 'offset', 0.2);
% disable modulation on output channel 1
m.disable_modulation(1)
```
</code-block>
</code-group>


