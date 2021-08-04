---
additional_doc: null
description: Set Arbitrary Waveform Generator to a default state
method: post
name: set_defaults
parameters: []
summary: set_defaults
---

<headers/>

Default state implies, 

- Enable all output channels
- Set output load to 1MOhm
- Set trigger source to Internal
- Sync phase accumulator of all output waveforms


::: tip INFO
A new reference to any instrument class will always be in its default state.
:::


<parameters/>

Usage in clients, 

<code-group>
<code-block title="Python">
```python
from moku.instruments import ArbitraryWaveformGenerator
i = ArbitraryWaveformGenerator('192.168.###.###', force_connect=False)
# AWG reference i is in default state
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuArbitraryWaveformGenerator('192.168.###.###', true);
% AWG reference m is in default state
```
</code-block>
</code-group>