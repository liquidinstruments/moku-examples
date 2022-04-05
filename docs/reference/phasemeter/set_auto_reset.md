---
additional_doc: null
description: Sets the auto-reset function on the phasemeter
method: post
name: set_auto_reset
available_on: "mokupro"
parameters:
- default: null
  description: Phase reset point
  name: value
  param_range: Off, 1pi, 2pi, 4pi
  type: string
  unit: null
- default: true
  description: Disable all implicit conversions and coercions.
  name: strict
  param_range: null
  type: boolean
  unit: null
summary: set_auto_reset
---


<headers/>
<parameters/>

### Examples

<code-group>
<code-block title="Python">
```python
from moku.instruments import Phasemeter
i = Phasemeter('192.168.###.###')
i.set_auto_reset(value='2pi')
```
</code-block>

<code-block title="MATLAB">
```matlab{8}
i = MokuPhasemeter('192.168.###.###');
i.set_auto_reset('2pi');
```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"value": "2pi"}'\
        http://<ip>/api/phasemeter/set_auto_reset
```
</code-block>

</code-group>

### Sample response
```json
{'Auto-reset': '±2π'}
```