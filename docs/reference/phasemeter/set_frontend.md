---
additional_doc: null
description: Configures the input impedance, coupling, and range for each channel
method: post
name: set_frontend
summary: set_frontend
available_on: "mokupro"

parameters:
- default: null
  description: Target channel
  name: channel
  param_range: 1, 2, 3, 4
  type: integer
  unit: null
- default: 1MOhm
  description: Impedance
  name: impedance
  param_range: 50Ohm, 1MOhm
  type: string
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
  param_range: 400mVpp, 4Vpp, 40Vpp
  type: string
  unit: null
- default: true
  description: Disable all implicit conversions and coercions.
  name: strict
  param_range: null
  type: boolean
  unit: null

---





<headers/>
<parameters/>


### Examples

<code-group>
<code-block title="Python">
```python
from moku.instruments import MokuPhasemeter
i = MokuPhasemeter('192.168.###.###', force_connect=False)
i.set_frontend(1, "DC", "40Vpp")
```
</code-block>

<code-block title="MATLAB">
```matlab
i = MokuPhasemeter('192.168.###.###', true);
i.set_frontend(1, "DC", "40Vpp");
```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"channel": 1, "coupling": "AC", "impedance": "1MOhm", "range": "4Vpp"}'\
        http://<ip>/api/phasemeter/set_frontend
```
</code-block>

</code-group>