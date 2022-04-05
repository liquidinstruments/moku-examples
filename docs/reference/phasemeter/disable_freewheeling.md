---
additional_doc: null
description: Disable free wheeling.
method: post
name: disable_freewheeling
parameters:
- default: true
  description: Disable freewheeling
  name: disable
  param_range: null
  type: boolean
  unit: null
- default: true
  description: Disable all implicit conversions and coercions.
  name: strict
  param_range: null
  type: boolean
  unit: null
summary: disable_freewheeling
available_on: "mokupro"
---

<headers/>
<parameters/>

### Examples

<code-group>
<code-block title="Python">
```python
from moku.instruments import Phasemeter

i = Phasemeter('192.168.###.###')
i.disable_freewheeling(disable=True)
```
</code-block>

<code-block title="MATLAB">
```matlab
i = MokuPhasemeter('192.168.###.###');
i.disable_freewheeling('disable',true);
```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"disable": true}'\
        http://<ip>/api/phasemeter/disable_freewheeling
```
</code-block>

</code-group>

### Sample response
```json
{"Freewheeling":"true"}
```
