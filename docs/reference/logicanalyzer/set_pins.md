---
additional_doc: null
description: Set the state of the digital I/O pins
method: post
name: set_pins
parameters:

- default: null
  description: List of pins with corresponding states to configure
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

### Sample request,
```json
[
  {
    "pin": 1,
    "state": "O"
  },
  {
    "pin": 2,
    "state": "H"
  },
  {
    "pin": 3,
    "state": "L"
  }
]
```
### Examples


<code-group>
<code-block title="Python">
```python
from moku.instruments import LogicAnalyzer
i = LogicAnalyzer('192.168.###.###')
pins = [dict(pin=1, state="O"),
        dict(pin=2, state="H"),
        dict(pin=3, state="L")]
i.set_pins(pins)
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuLogicAnalyzer('192.168.###.###');
% Configure pin1 to "O"
pin1.pin = 1;
pin1.state = 'O';

% Configure pin2 to "H"
pin2.pin = 2;
pin2.state = 'H';

% Configure pin3 to "L"
pin3.pin = 3;
pin3.state = 'L';

m.set_pins([pin1,pin2,pin3]);
```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: 17ee67f8476'\
        -H 'Content-Type: application/json'\
        --data '[{"pin": 1, "state": "O"}, {"pin": 2, "state": "H"}, {"pin": 3, "state": "L"}]'\
        http://<ip>/api/logicanalyzer/set_pins
```
</code-block>

</code-group>

:::tip Note
set_pin only configures the state of the Pin, to generate a pattern on a pin use
[generate_pattern](generate_pattern.md)
:::


### Sample response,
```json
[
  "O", "H", "L"
]
```
