---
additional_doc: null
description: Returns a short summary of current instrument state
method: get
name: summary
parameters: []
summary: summary
---

<headers/>
<parameters/>

<code-group>
<code-block title="Python">
```python{5}
from moku.instruments import MokuPhasemeter
i = MokuPhasemeter('192.168.###.###', force_connect=False)
print(i.summary())
```
</code-block>

<code-block title="MATLAB">
```matlab{5}
m = MokuPhasemeter('192.168.###.###', true);
disp(m.summary())
```
</code-block>
</code-group>

Sample response,

```text
    'Moku:Pro Phasemeter
     Input 1 - DC coupling, 1 MOhm impedance, 4 Vpp range, 2.5 kHz bandwidth
     Input 2 - DC coupling, 1 MOhm impedance, 4 Vpp range, 2.5 kHz bandwidth
     Input 3 - DC coupling, 1 MOhm impedance, 4 Vpp range, 2.5 kHz bandwidth
     Input 4 - DC coupling, 1 MOhm impedance, 4 Vpp range, 2.5 kHz bandwidth
     Acquisition rate: 1.4901161194e+02 Hz
     Internal 10 MHz clock'
```