---
additional_doc: null
description: Set the state of the digital I/O pins
method: post
name: set_pins
parameters:

- default: null
  description: List of pins with corresponding state to configure
  name: pins
  type: array
  unit: null
  param_range: null
- default: true
  description: Disable all implicit conversions and coercions. 
  name: strict
  param_range: null
  type: boolean
  unit: null
summary: set_pins
available_on: "mokugo"
---


<headers/>
<parameters/>

Please refer to [Pin Status Definitions](README.md) for the list of available statuses

:::tip Note
set_pin only configures the state of the Pin, to generate a pattern on a pin use
[set_pattern_generator](set_pattern_generator.md)
:::


### Sample request,
```json
[
  {
    "pin": 1,
    "state": "PG1"
  },
  {
    "pin": 2,
    "state": "PG2"
  }
]
```

### Examples


<code-group>
<code-block title="Python">
```python
from moku.instruments import LogicAnalyzer
i = LogicAnalyzer('192.168.###.###')
pins = [dict(pin=1, state="PG1"),
        dict(pin=2, state="PG2")]
i.set_pins(pins)
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuLogicAnalyzer('192.168.###.###');
% Configure Pin to corresponding states
pin_status = [struct('pin', 1, 'state', 'PG1'),...
    struct('pin', 2, 'state', 'PG2')];
m.set_pins(pin_status);
```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: 17ee67f8476'\
        -H 'Content-Type: application/json'\
        --data '[{"pin": 1, "state": "PG1"}, {"pin": 2, "state": "PG2"}]'\
        http://<ip>/api/logicanalyzer/set_pins
```
</code-block>

</code-group>

