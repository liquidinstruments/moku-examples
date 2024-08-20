---
additional_doc:
    NOTE - Interface currently only supports using the first two DFB loops which are
    paired with the first two ADC channels, e.g. Moku:Pro's third and fourth loops and ADC3, ADC4
    cannot yet be used through this API.
description: Set the linear combination of ADC input signals for a given DFB channel.
method: post
name: set_control_matrix
parameters:
    - default: null
      description: Target DFB channel
      name: channel
      param_range: 1, 2
      type: integer
      unit: null
    - default: null
      description: ADC Input 1 gain for this channel
      name: input_gain1
      param_range: -20 to 20
      type: number
      unit: dB
    - default: null
      description: ADC Input 1 gain for this channel
      name: input_gain2
      param_range: -20 to 20
      type: number
      unit: dB
    - default: true
      description: Disable all implicit conversions and coercions.
      name: strict
      param_range: null
      type: boolean
      unit: null
summary: set_control_matrix
---

<headers/>
<parameters/>

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import DigitalFilterBox
i = DigitalFilterBox('192.168.###.###')
i.set_control_matrix(1, input_gain1=1, input_gain2=0)
i.set_control_matrix(2, input_gain1=0, input_gain2=1)
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuDigitalFilterBox('192.168.###.###');
m.set_control_matrix(1, 1, 0);
m.set_control_matrix(2, 0, 1);
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"channel": 1, "input_gain1": 1, "input_gain2": -1}'\
        http://<ip>/api/digitalfilterbox/set_control_matrix
```

</code-block>

</code-group>

### Sample response

```json
{
    "input_gain1": 1.0,
    "input_gain2": 0.0
}
```
