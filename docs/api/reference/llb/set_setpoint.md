---
additional_doc: null
description: Configure the the desired set point
method: post
name: set_setpoint
parameters:
    - default: null
      description: setpoint voltage
      name: setpoint
      param_range: -1 to 1
      type: integer
      unit: V
    - default: true
      description: Disable all implicit conversions and coercions.
      name: strict
      param_range: null
      type: boolean
      unit: null
summary: set_setpoint
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
i.set_setpoint(1)

```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuLaserLockBox('192.168.###.###');
m.set_aux_oscillator()
% Set the set point to 1V
m.set_setpoint(1)
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"setpoint": 1}'\
        http://<ip>/api/laserlockbox/set_setpoint
```

</code-block>

</code-group>

### Sample response

```json
{
    "setpoint": 1
}
```
