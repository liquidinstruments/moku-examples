---
additional_doc: null
description: Set output signal gain
method: post
name: set_output_gain
parameters:
    - default: null
      description: Target channel
      name: channel
      param_range:
          mokupro: 1, 2, 3, 4
          mokudelta: 1, 2, 3, 4
      type: integer
      unit: null
    - default: null
      description: Output gain
      name: gain
      param_range: -40 to 40
      type: number
      unit: dB
    - default: true
      description: Disable all implicit conversions and coercions.
      name: strict
      param_range: null
      type: boolean
      unit: null
summary: set_output_gain
---

<headers/>
<parameters/>

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import DigitalFilterBox
i = DigitalFilterBox('192.168.###.###')
# Set output gain to 0dB
i.set_output_gain(1, 0)
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuDigitalFilterBox('192.168.###.###');
% Set output gain to 0dB
m.set_output_gain(1, 0);
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"channel": 1, "gain": 0}'\
        http://<ip>/api/digitalfilterbox/set_output_gain
```

</code-block>

</code-group>

### Sample response

```json
{
    "gain": 5.0
}
```
