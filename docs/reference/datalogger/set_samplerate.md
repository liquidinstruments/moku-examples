---
additional_doc: null
description: Sets the sample rate of the instrument
method: post
name: set_samplerate
parameters:
- default: null
  description: Target samples per second
  name: sample_rate
  param_range: 10 to 1e6
  type: number
  unit: null
- default: true
  description: Disable all implicit conversions and coercions.
  name: strict
  param_range: null
  type: boolean
  unit: null
summary: set_samplerate
---


<headers/>
<parameters/>

Usage in clients, 

<code-group>
<code-block title="Python">
```python{5}
from moku.instruments import Datalogger
i = Datalogger('192.168.###.###', force_connect=False)
# Generate Sine wave on Output1
# Set required sample rate
i.set_samplerate(1e3)
```
</code-block>

<code-block title="MATLAB">
```matlab{8}
m = MokuDatalogger('192.168.###.###', true);
% Generate a sine wave on Channel 1
# Set required sample rate
m.set_samplerate(1e3);
```
</code-block>
</code-group>