---
additional_doc: null
description: Get current state of a Digital I/O pin
deprecated: true
deprecated_msg: get_pin is deprecated, use [get_pin_mode](./getters.html) instead.
method: post
name: get_pin
summary: get_pin
parameters:
    - default: null
      description: Target pin to get the state
      name: pin
      param_range: 1 to 16
      type: integer
      unit: null
---

<headers/>
<parameters/>

#### Available states for a pin are

| State | Description           |
| ----- | :-------------------- |
| I     | Input                 |
| O     | Output                |
| H     | High, pin is set to 1 |
| L     | Low, pin is set to 0  |
| X     | Off, Pin is off       |

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import LogicAnalyzer
i = LogicAnalyzer('192.168.###.###')
i.get_pin()
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuLogicAnalyzer('192.168.###.###');
m.get_pin();
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"pin":1}'\
        http://<ip>/api/logicanalyzer/get_pin
```

</code-block>

</code-group>

### Sample response

```json
{
    "state": "H"
}
```
