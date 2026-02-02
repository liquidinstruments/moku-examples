---
additional_doc: null
description: Reset the Lock-in Amplifier to its default state
method: post
name: set_defaults
parameters: []
summary: set_defaults
---

<headers/>
<parameters/>

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import LockInAmp
i = LockInAmp('192.168.###.###')
i.set_defaults()
# lockinamp reference i is in default state
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuLockInAmp('192.168.###.###');
m.set_defaults();
% lockinamp reference m is in default state
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{}'\
        http://<ip>/api/lockinamp/set_defaults
```

</code-block>
</code-group>
