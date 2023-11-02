---
additional_doc: Analog signals are converted to digital using the thresholds provided.
description: Sets the threshold voltages of analog inputs
method: post
name: set_analog_mode
parameters:
- default: 1.25
  description: High threshold for analog inputs
  name: high
  type: float
  unit: V
- default: 0.75
  description: Low threshold for analog inputs
  name: low
  type: float
  unit: V
- default: true
  description: Disable all implicit conversions and coercions.
  name: strict
  param_range: null
  type: boolean
  unit: null
summary: set_analog_mode
---

<headers/>
<parameters/>

### Examples

<code-group>
<code-block title="Python">
```python
from moku.instruments import LogicAnalyzer
i = LogicAnalyzer('192.168.###.###')
i.set_analog_mode(high=1,low=0.75)
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuLogicAnalyzer('192.168.###.###');
m.set_analog_mode('high', 1, 'low', '0.75');
```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"high":1, "low":0.75}'\
        http://<ip>/api/logicanalyzer/set_analog_mode
```
</code-block>

</code-group>

### Sample response
```json
{"high":1.0, "low":0.75}
```
