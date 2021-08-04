---
additional_doc: When using either of the clients, user can access this function directly       from            instrument reference.
description: Get serial of the Moku
method: get
name: serial_number
parameters: []
summary: serial_number
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