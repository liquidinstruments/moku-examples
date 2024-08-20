---
additional_doc: null
description: Configures the limit on desired output channel
method: post
name: set_output_limit
parameters:
    - default: null
      description: Target output channel to configure
      name: channel
      param_range:
          mokugo: 1, 2
          mokulab: 1, 2
          mokupro: 1, 2, 3, 4
      type: integer
      unit: null
    - default: False
      description: Enable/Disable voltage limiter
      name: enable
      param_range: null
      type: boolean
      unit: null
    - default: nullptr
      description: Low voltage limit
      name: low_limit
      param_range: -5 to 5
      type: integer
      unit: V
    - default: null
      description: High voltage limit
      name: high_limit
      param_range: -5 to 5
      type: integer
      unit: V
    - default: true
      description: Disable all implicit conversions and coercions.
      name: strict
      param_range: null
      type: boolean
      unit: null
summary: set_output_limit
---

<headers/>
<parameters/>

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import LaserLockBox
i = LaserLockBox('192.168.###.###')
i.set_output_limit(channel=3, high_limit=1)
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuLaserLockBox('192.168.###.###');
m.set_output_limit(1, 'high_limit', 1);
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"channel": 1, "high_limit": 2}'\
        http://<ip>/api/laserlockbox/set_output_limit
```

</code-block>

</code-group>
