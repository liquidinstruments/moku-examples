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

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import Datalogger
i = Datalogger('192.168.###.###')
print(i.summary())
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuDatalogger('192.168.###.###');
disp(m.summary())
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        http://<ip>/api/datalogger/summary
```

</code-block>

</code-group>

### Sample response

```plaintext
Moku:Go Data Logger
Input 1, DC coupling, 10 Vpp range
Input 2, DC coupling, 10 Vpp range
Acquisition rate: 1.0000000000e+03 Hz, Precision mode
Output 1 (off) - Sine, 10.000 000 000 000 MHz, 1.000 0 Vpp, offset 0.000 0 V, phase 0.000 000 deg
Output 2 (off) - Ramp, 50.000 000 000 kHz, 500.0 mVpp, offset 0.000 0 V, phase 0.000 000 deg, symmetry 90.00 %
```
