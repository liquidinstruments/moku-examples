---
additional_doc: null
description: Generate pattern on a single channel
method: post
name: generate_pattern
parameters:
- default: null
  description: Pin to generate pattern on
  name: pin
  param_range: 1 to 16
  type: integer
  unit: null
- default: null
  description: Pattern to generate, array filled with 0's and 1's. Maximum size is 1024
  name: pattern
  param_range: null
  type: array
  unit: null
- default: 1
  description: Divider to scale down the base frequency of 125 MHz to the tick frequency.
    For example, a divider of 2 provides a 62.5 MHz tick frequency.
  name: divider
  param_range: 1 to 1e6
  type: integer
  unit: null
- default: true
  description: Repeat forever
  name: repeat
  param_range: null
  type: boolean
  unit: null
- default: 1
  description: Number of iterations, valid when repeat is set to false
  name: iterations
  param_range: 1 to 8192
  type: integer
  unit: null
- default: true
  description: Disable all implicit conversions and coercions.
  name: strict
  param_range: null
  type: boolean
  unit: null
summary: generate_pattern
available_on: "mokugo"
---

<headers/>
<parameters/>

:::tip Note
A pattern can be generated only when the state of Pin is set to 'O'. 
Refer [Pin Status Definitions](README.md) for the list of available statuses
:::

### Examples


<code-group>
<code-block title="Python">
```python
from moku.instruments import LogicAnalyzer
i = LogicAnalyzer('192.168.###.###')
i.set_pin(1, "O")
i.set_pin(2, "H")
i.set_pin(3, "L")
# Configure the output pattern for Pin 1
i.generate_pattern(pin=1, pattern=[1, 0, 0, 0, 0, 0, 0, 0])
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuLogicAnalyzer('192.168.###.###');
m.set_pin('pin',1, 'state',"O");
m.set_pin('pin',2, 'state', "H");
m.set_pin('pin',3, 'state', "L");
% Configure the output pattern on Pin 8 to [1 1 0 0]
m.generate_pattern('pin',1,'pattern', [1 1 0 0]);
```
</code-block>

<code-block title="cURL">
```bash
# If the pattern is longer, consider putting the data in a JSON file
# rather than passing on the command line
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"pin": 1, "pattern": [1, 1, 0, 0]}'\
        http://<ip>/api/logicanalyzer/generate_pattern
```
</code-block>

</code-group>

### Sample response,

```json
{
  "divider": 1,
  "iterations": 1,
  "repeat": true
}
```