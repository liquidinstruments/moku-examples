---
additional_doc: null
description: Set multiple control register values.
method: post
name: set_controls
parameters:
    - default: null
      description: List of maps of control ID and value
      name: controls
      type: array
      unit: null
    - default: true
      description: Disable all implicit conversions and coercions.
      name: strict
      param_range: null
      type: boolean
      unit: null
summary: set_controls
---

<headers/>
<parameters/>

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import MultiInstrument, CloudCompile
m = MultiInstrument('192.168.###.###', platform_id=2)
cc = m.set_instrument(1, CloudCompile)
# set connections
controls = {"controls": [
    {"idx": 0, "value": 32836},
    {"idx": 1, "value": 450},
    {"idx": 2, "value": 32},
    {"idx": 3, "value": 2048},
    {"idx": 4, "value": 2147450879}
]}
cc.set_controls(controls)
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuMultiInstrument('192.168.###.###', 'platform_id', 2);
cc = m.set_instrument(1, MokuCloudCompile);
% set connections
% TODO
```

</code-block>

<code-block title="cURL">

```bash
$: cat controls.json
{
  "controls":
  [
    {"idx":0,"value":32836},
    {"idx":1,"value":450},
    {"idx":2,"value":32},
    {"idx":3,"value":2048},
    {"idx":4,"value":2147450879}
  ]
}
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data @controls.json\
        http://<ip>/api/slot1/cloudcompile/set_controls
```

</code-block>

</code-group>
