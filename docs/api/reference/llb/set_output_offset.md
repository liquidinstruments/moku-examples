---
additional_doc: null
description: Set output signal offset
method: post
name: set_output_offset
parameters:
    - default: null
      description: Target channel
      name: channel
      param_range: 1, 2
      type: integer
      unit: null
    - default: null
      description: Output DC offset
      name: offset
      param_range:
          mokugo: -2.5 to 2.5
          mokulab: -1 to 1
          mokupro: -1 to 1
          mokudelta: -500e-3 to 500e-3
      type: number
      unit: V
    - default: true
      description: Disable all implicit conversions and coercions.
      name: strict
      param_range: null
      type: boolean
      unit: null
summary: set_output_offset
---

<headers/>
<parameters/>

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import LaserLockBox
i = LaserLockBox('192.168.###.###', force_connect=True)
# Set Channel 1 output offset to 2 V
i.set_output_offset(1, offset=2)
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuLaserLockBox('192.168.###.###', force_connect=true);
% Set Channel 1 output offset to 2 V
m.set_output_offset(1, 'offset', 2);
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"channel": 1, "offset": 1}'\
        http://<ip>/api/laserlockbox/set_output_offset
```

</code-block>

</code-group>

### Sample response

```json
{ "offset": 2.0 }
```
