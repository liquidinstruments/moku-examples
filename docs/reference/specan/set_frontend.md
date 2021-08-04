---
additional_doc: null
description: Configures the input impedance, coupling, and range for each channel.
method: post
name: set_frontend
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
  description: Input Coupling
  name: coupling
  param_range: AC, DC
  type: string
  unit: null
- default: null
  description: Input Range
  name: range
  param_range: 10Vpp, 50Vpp
  type: string
  unit: null
- default: true
  description: Disable all implicit conversions and coercions.
  name: strict
  param_range: null
  type: boolean
  unit: null
summary: set_frontend
---





<headers/>
<parameters/>

Usage in clients, 

<code-group>
<code-block title="Python">
```python{3}
from moku.instruments import SpectrumAnalyzer
i = SpectrumAnalyzer('192.168.###.###', force_connect=False)
i.set_defaults(1, "AC", "50Vpp")
```
</code-block>

<code-block title="MATLAB">
```matlab{2}
m = MokuSpectrumAnalyzer('192.168.###.###', true);
m.set_defaults(1, 'AC', '50Vpp')
```
</code-block>
</code-group>