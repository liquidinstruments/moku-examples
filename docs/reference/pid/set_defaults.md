---
additional_doc: null
description: Set the PID Controller to its default state.
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
from moku.instruments import PIDController
i = PIDController('192.168.###.###')
i.set_defaults()
# PIDController reference i is in default state
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuPIDController('192.168.###.###');
m.set_defaults();
% PIDController reference m is in default state
```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{}'\
        http://<ip>/api/pidcontroller/set_defaults
```
</code-block>

</code-group>