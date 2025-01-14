---
additional_doc: null
deprecated: true
deprecated_msg: This method is deprecated and will be removed soon. Please use **hysteresis** parameter in [set_trigger](./set_trigger.html)
description: Configures the hysteresis around trigger point.
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
from moku.instruments import DigitalFilterBox
i = DigitalFilterBox('192.168.###.###')
# Set instrument to desired state
i.set_hysteresis("Absolute",1)
```

</code-block>

<code-block title="MATLAB">

```matlab
i = MokuDigitalFilterBox('192.168.###.###');
% Set instrument to desired state
i.set_hysteresis('Absolute',1);
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"hysteresis_mode": "Absolute", "value": 1}'\
        http://<ip>/api/digitalfilterbox/set_hysteresis
```

</code-block>

</code-group>

### Sample response

```json
{
    "hysteresis_mode": "Absolute",
    "value": 1.0
}
```
