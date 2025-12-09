---
additional_doc: null
description: Set the Laser Lock Box to to its default state.
method: post
name: set_defaults
parameters: []
summary: set_defaults
---

<headers/>
<parameters/>

::: tip INFO
Reference to any instrument object will always be in default state.
:::

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import LaserLockBox
i = LaserLockBox('192.168.###.###', force_connect=True)
i.set_defaults()
# LaserLockBox reference i is in default state
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuLaserLockBox('192.168.###.###', force_connect=true);
m.set_defaults();
% LaserLockBox reference m is in default state
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{}'\
        http://<ip>/api/laserlockbox/set_defaults
```

</code-block>

</code-group>
