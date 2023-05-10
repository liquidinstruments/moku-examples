---
additional_doc: null
description: Configures the input impedance, coupling, and range for each input channel.
method: post
name: set_frontend
parameters:
- default: null
  description: Target channel
  name: channel
  param_range:
   mokugo: 1, 2
   mokulab: 1, 2
   mokupro: 1, 2, 3, 4
  type: integer
  unit: null
- default: 1MOhm
  description: Impedance
  name: impedance
  param_range:
   mokugo: 1MOhm
   mokulab: 50Ohm, 1MOhm
   mokupro: 50Ohm, 1MOhm
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
  param_range: 
   mokugo: 10Vpp, 50Vpp
   mokulab: 1Vpp, 10Vpp
   mokupro: 400mVpp, 4Vpp, 40Vpp
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


### Examples

<code-group>
<code-block title="Python">
```python
from moku.instruments import FrequencyResponseAnalyzer
i = FrequencyResponseAnalyzer('192.168.###.###')
i.set_frontend(1, "1MOhm", "AC", "10Vpp")
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuFrequencyResponseAnalyzer('192.168.###.###');
m.set_frontend(1, '1MOhm', 'DC', '10Vpp');
```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"channel": 1, "impedance": "1MOhm", "coupling": "DC", "range": "10Vpp"}'\
        http://<ip>/api/fra/set_frontend
```
</code-block>
</code-group>

### Sample response
```json
{
 "coupling":"DC",
 "impedance":"1MOhm",
 "range":"10Vpp"
}
```