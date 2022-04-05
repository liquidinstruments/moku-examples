---
additional_doc: null
description: Configures the hysteresis around trigger point.
method: post
name: set_hysteresis
parameters:
- default: null
  description: Trigger sensitivity hysteresis mode
  name: hysteresis_mode
  param_range: Absolute, Relative
  type: string
  unit: null
- default: 0
  description: Hysteresis around trigger
  name: value
  param_range: null
  type: number
  unit: null
- default: true
  description: Disable all implicit conversions and coercions.
  name: strict
  param_range: null
  type: boolean
  unit: null
summary: set_hysteresis
group: Monitors
---

<headers/>
<parameters/>

### Examples

<code-group>
<code-block title="Python">
```python
from moku.instruments import LockInAmp
i = LockInAmp('192.168.###.###')
i.set_hysteresis("Absolute",5)
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuLockInAmp('192.168.###.###');
m.set_hysteresis('hysteresis_mode', 'Absolute', 'value', 5);
```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"hysteresis_mode": "Absolute", "value": 5}'\
        http://<ip>/api/lockinamp/set_hysteresis
```
</code-block>

</code-group>

### Sample response
```json
{"Trigger hysteresis":5,"Trigger hysteresis mode":"Absolute"}
```
