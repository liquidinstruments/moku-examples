---
additional_doc: null
description: Enable trigger during the rising portion of scan waveform.
method: post
name: enable_conditional_trigger
parameters:
- default: true
  description: Enable/Disable conditional trigger
  name: enable
  param_range: null
  type: boolean
  unit: null
- default: true
  description: Disable all implicit conversions and coercions.
  name: strict
  param_range: null
  type: boolean
  unit: null
summary: enable_conditional_trigger
mark_as_beta: true
---

<headers/>
<parameters/>

### Examples

<code-group>
<code-block title="Python">
```python
from moku.instruments import LaserLockBox
i = LaserLockBox('192.168.###.###')
i.enable_conditional_trigger()
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuLaserLockBox('192.168.###.###');
m.enable_conditional_trigger();
```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{}'\
        http://<ip>/api/laserlockbox/enable_conditional_trigger
```
</code-block>

</code-group>
