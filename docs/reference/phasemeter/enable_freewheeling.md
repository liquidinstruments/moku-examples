---
additional_doc: null
description: Enable free wheeling.
method: post
name: enable_freewheeling
parameters:
- default: true
  description: Enable freewheeling
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
summary: enable_freewheeling
---

<headers/>
<parameters/>

### Examples

<code-group>
<code-block title="Python">
```python
from moku.instruments import Phasemeter

i = Phasemeter('192.168.###.###')
i.enable_freewheeling()
```
</code-block>

<code-block title="MATLAB">
```matlab
i = MokuPhasemeter('192.168.###.###');
i.enable_freewheeling();
```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"disable": true}'\
        http://<ip>/api/phasemeter/enable_freewheeling
```
</code-block>

</code-group>

