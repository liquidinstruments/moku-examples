---
additional_doc: null
description: Configures the input impedance, coupling, and range for each channel
method: post
name: set_frontend
parameters:
- default: null
  description: Target channel
  name: channel
  param_range:
   mokugo: 1, 2
   mokupro: 1, 2, 3, 4
  type: integer
  unit: null
- default: 1MOhm
  description: Impedance
  name: impedance
  param_range:
   mokugo: 1MOhm
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
  description: Input attenuation
  name: attenuation
  param_range: 
    mokugo: 0dB, -14dB
    mokupro: -20dB, -40dB
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
from moku.instruments import LockInAmp
i = LockInAmp('192.168.###.###')
i.set_frontend(1, impedance="1MOhm", coupling="AC", attenuation="14dB")
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuLockInAmp('192.168.###.###');
m.set_frontend(1, 'DC', '1MOhm', '14dB');
```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"channel": 1, "impedance": "1MOhm", "coupling": "AC", "attenuation": "14dB"}'\
        http://<ip>/api/lockinamp/set_frontend
```
</code-block>

</code-group>

### Sample response
```json
{
  "attenuation": "14dB",
  "coupling": "AC",
  "impedance": "1MOhm"
}
```