---
additional_doc: null
description: Sets the acquisition speed of the instrument
method: post
name: set_acquisition_speed
parameters:
- default: null
  description: null
  name: speed
  param_range: 30Hz, 120Hz, 480Hz, 2kHz, 15kHz, 122kHz
  type: number
  unit: null
- default: true
  description: Disable all implicit conversions and coercions.
  name: strict
  param_range: null
  type: boolean
  unit: null
summary: set_acquisition_speed
available_on: "mokupro"
---


<headers/>
<parameters/>

Usage in clients, 

<code-group>
<code-block title="Python">
```python{4}
from moku.instruments import MokuPhasemeter
i = MokuPhasemeter('192.168.###.###', force_connect=False)
# Sets the acquisition speed to 480 Hz
i.set_acquisition_speed(speed='480Hz')
```
</code-block>

<code-block title="MATLAB">
```matlab{3}
i = MokuPhasemeter('192.168.###.###', true);
# Set required acquisition speed
i.set_acquisition_speed('480Hz');
```
</code-block>
</code-group>