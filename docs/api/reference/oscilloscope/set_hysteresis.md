---
additional_doc: null
description: Configures the hysteresis around trigger point.
deprecated: true
deprecated_msg: set_hysteresis is deprecated, use `hysteresis` parameter of [set_trigger](./set_trigger.md) instead.
method: post
name: set_hysteresis
parameters:
    - default: null
      description: Trigger sensitivity hysteresis mode
      name: hysteresis_mode
      param_range: Absolute, Relative
      type: string
      unit: null
    - default: 0
      description: Hysteresis around trigger
      name: value
      param_range: null
      type: number
      unit: null
    - default: true
      description: Disable all implicit conversions and coercions.
      name: strict
      param_range: null
      type: boolean
      unit: null
summary: set_hysteresis
---

<headers/>
<parameters/>

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import Oscilloscope
i = Oscilloscope('192.168.###.###')
i.set_hysteresis("Absolute",0.1)
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuOscilloscope('192.168.###.###');
m.set_hysteresis('hysteresis_mode',"Absolute","value",0.1);
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"hysteresis_mode":"Absolute","value":0.1}'\
        http://<ip>/api/oscilloscope/set_hysteresis
```

</code-block>

</code-group>

### Sample response

```json
{
    "hysteresis_mode": "Absolute",
    "value": 0.1
}
```
