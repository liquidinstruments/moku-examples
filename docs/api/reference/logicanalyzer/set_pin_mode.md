---
additional_doc: This method is unavailable for Moku:Pro and Moku:Lab because
    these models do not have digital IO port.
description: Sets the state of Moku:Go and MokuLDelta digital IO to either input or output
method: post
name: set_pin_mode
parameters:
    - default: null
      description: Target pin to configure
      name: pin
      param_range: 1 to 16
      type: integer
      unit: null
    - default: null
      description: State of the target pin.
      name: state
      param_range: X, I, PG1, PG2
      type: string
      unit: null
    - default: true
      description: Disable all implicit conversions and coercions.
      name: strict
      param_range: null
      type: boolean
      unit: null
summary: set_pin_mode
---

<headers/>
<parameters/>

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import LogicAnalyzer
i = LogicAnalyzer('192.168.###.###')
i.set_pin_mode(pin=1,state="PG1")
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuLogicAnalyzer('192.168.###.###');
m.set_pin_mode('pin', 1, 'state', 'PG1');
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"pin":1, "state":"PG1"}'\
        http://<ip>/api/logicanalyzer/set_pin_mode
```

</code-block>

</code-group>

### Sample response

```json
{ "pin": "pin1", "state": "PG1" }
```
