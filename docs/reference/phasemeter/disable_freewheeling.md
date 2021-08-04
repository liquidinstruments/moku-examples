---
additional_doc: null
description: Disable free wheeling.
method: post
name: disable_freewheeling
parameters:
- default: true
  description: Disable freewheeling
  name: disable
  param_range: null
  type: boolean
  unit: null
- default: true
  description: Disable all implicit conversions and coercions.
  name: strict
  param_range: null
  type: boolean
  unit: null
summary: disable_freewheeling
available_on: "mokupro"
---

<headers/>
<parameters/>

Usage in clients, 

<code-group>
<code-block title="Python">
```python
from moku.instruments import ArbitraryWaveformGenerator

i = ArbitraryWaveformGenerator('192.168.###.###', force_connect=False)
# Configure the output waveform in each channel
i.enable_output(1)
i.enable_output(2, false)
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuArbitraryWaveformGenerator('192.168.###.###', true);
% Configure the output waveform in each channel
m.enable_output(1);
m.enable_output(2, 'enable','false');
```
</code-block>
</code-group>

