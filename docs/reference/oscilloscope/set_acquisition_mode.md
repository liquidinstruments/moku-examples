---
additional_doc: null
description: Set acquisition mode
method: post
name: set_acquisition_mode
parameters:
- default: Normal
  description: Acquisition Mode
  name: mode
  param_range: Normal, Precision, PeakDetect
  type: string
  unit: null
- default: true
  description: Disable all implicit conversions and coercions.
  name: strict
  param_range: null
  type: boolean
  unit: null
summary: set_acquisition_mode
---

<headers/>
<parameters/>

Usage in clients, 

<code-group>
<code-block title="Python">
```python
from moku.instruments import Oscilloscope
i = Oscilloscope('192.168.###.###', force_connect=False)
i.set_acquisition_mode(mode="Precision")
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuOscilloscope('192.168.###.###', true);
m.set_acquisition_mode('mode', 'Precision')
```
</code-block>
</code-group>