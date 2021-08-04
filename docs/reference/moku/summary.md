---
additional_doc: null
description: Summary of the current state
method: get
name: summary
parameters: []
summary: summary
---




When using either of the clients, user can access this function directly from
instrument reference.

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


<headers/>
<parameters/>