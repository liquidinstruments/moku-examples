---
additional_doc: null
description: Get current state of all Digital I/O pins
deprecated: true
deprecated_msg: get_pins is deprecated, use [get_pin_mode](./getters.md) instead.
method: post
name: get_pins
parameters: []
summary: get_pins
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
i.get_pins()
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuLogicAnalyzer('192.168.###.###');
m.get_pins();
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{}'\
        http://<ip>/api/logicanalyzer/get_pins
```

</code-block>

</code-group>

### Sample response

```json
["O", "H", "L", "I", "I", "I", "I", "I", "I", "I", "I", "I", "I", "I", "I", "I"]
```
