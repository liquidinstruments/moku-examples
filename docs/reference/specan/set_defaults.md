---
additional_doc: null
description: Set the Spectrum Analyzer to its default state.
method: post
name: set_defaults
parameters: []
summary: set_defaults
---

<headers/>


Default state implies,

- Enable both input channels
- Set Input Coupling to DC
- Set Input Range to 10Vpp


::: tip INFO
Reference to any instrument object will always be in default state.
:::

<parameters/>

Usage in clients, 

<code-group>
<code-block title="Python">
```python
from moku.instruments import SpectrumAnalyzer
i = SpectrumAnalyzer('192.168.###.###', force_connect=False)
# SpectrumAnalyzer reference i is in default state
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuSpectrumAnalyzer('192.168.###.###', true);
% SpectrumAnalyzer reference m is in default state
```
</code-block>
</code-group>