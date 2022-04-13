---
additional_doc: null
description: Sets the left- and right-hand span for the time axis. Units are seconds
  relative to the trigger point.
method: post
name: set_timebase
parameters:
- default: null
  description: Time from the trigger point to the left of screen.
  name: t1
  param_range: null
  type: number
  unit: null
- default: null
  description: Time from the trigger point to the right of screen. (Must be a positive
    number, i.e. post trigger event)
  name: t2
  param_range: null
  type: number
  unit: null
- default: null
  description: Toggle Roll Mode
  name: roll_mode
  param_range: null
  type: boolean
  unit: null
- default: true
  description: Disable all implicit conversions and coercions.
  name: strict
  param_range: null
  type: boolean
  unit: null
summary: set_timebase
available_on: "mokugo"
---


<headers/>
<parameters/>

<code-group>
<code-block title="Python">
```python
from moku.instruments import LogicAnalyzer
i = LogicAnalyzer('192.168.###.###')
i.set_timebase(-0.5, 0, roll_mode=True)
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuLogicAnalyzer('192.168.###.###');
m.set_timebase(-0.5, 0, 'roll_mode', True);
```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"t1": -0.5, "t2": 0, "roll_mode": true}'\
        http://<ip>/api/logicanalyzer/set_timebase
```
</code-block>

</code-group>

### Sample response
```json
{
  "offset": 0.25,
  "roll_mode": true,
  "span": 0.5
}
```