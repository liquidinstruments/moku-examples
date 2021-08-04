---
additional_doc: null
description: Get current acquisition speed
method: get
name: get_acquisition_speed
parameters: []
summary: get_acquisition_speed
available_on: "mokupro"
---


<headers/>

Examples,

<code-group>
<code-block title="Python">
```python
from moku.instruments import ArbitraryWaveformGenerator

i = ArbitraryWaveformGenerator('192.168.###.###', force_connect=False)

# Here you can access the serial_number function
i.serial_number()
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuOscilloscope('192.168.###.###', false);

% Here you can access the serial_number function
m.serial_number()
```
</code-block>
</code-group>



<parameters/>