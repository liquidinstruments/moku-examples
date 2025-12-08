---
additional_doc: null
description: Configure the digital input gain to improve measurement precision. This gain setting is only available for Input 1.
method: post
name: set_digital_input_gain
parameters:
    - default: null
      description: Adds digital input gain
      name: digital_gain
      param_range: 0dB, +24dB, +48dB
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
i = LaserLockBox('192.168.###.###', force_connect=True)
# Set the digital input gain to 24 dB
i.set_digital_input_gain('+24dB')

```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuLaserLockBox('192.168.###.###', force_connect=true);
% Set the digital input gain to 24 dB
m.set_digital_input_gain('+24dB')
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"digital_gain": "+24dB"}'\
        http://<ip>/api/laserlockbox/set_digital_input_gain
```

</code-block>

</code-group>

### Sample response

```json
{
    "digital_gain": "+24dB"
}
```
