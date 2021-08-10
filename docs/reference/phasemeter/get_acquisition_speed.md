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
There are 6 different acquisition speeds available in the Phasemeter instrument, the response from the Moku matches the following list:

- -19 = 30 Hz
- -17 = 120 Hz
- -15 = 480 Hz
- -13 = 2 kHz
- -10 = 15 kHz
- -7 = 122 kHz

Examples,

<code-group>
<code-block title="Python">
```python
from moku.instruments import MokuPhasemeter

i = MokuPhasemeter('192.168.###.###', force_connect=False)

# Request the acquisition speed of the Phasemeter
speed = i.get_acquisition_speed()
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuPhasemeter('192.168.###.###', false);

% Request the acquisition speed of the Phasemeter
speed = m.get_acquisition_speed()
```
</code-block>
</code-group>



<parameters/>