---
additional_doc: null
description: Configure the the desired input gain
method: post
name: set_digital_input_gain
parameters:
    - default: null
      description: Input gain
      name: digital_gain
      param_range: 48dB, 24dB, 0dB
      type: string
      unit: 
    - default: true
      description: Disable all implicit conversions and coercions.
      name: strict
      param_range: null
      type: boolean
      unit: null
summary: set_digital_input_gain
---

<headers/>

<parameters/>

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import LaserLockBox
i = LaserLockBox('192.168.###.###')
i.set_aux_oscillator()
# Set the set point to 1V
i.set_digital_input_gain('0dB')

```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuLaserLockBox('192.168.###.###');
m.set_aux_oscillator()
% Set the set point to 1V
m.set_digital_input_gain('0dB')
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"digital_gain": "0dB"}'\
        http://<ip>/api/laserlockbox/set_digital_input_gain
```

</code-block>

</code-group>

### Sample response

```json
{
    "digital_gain": "0dB"
}
```
