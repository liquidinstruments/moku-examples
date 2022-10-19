---
additional_doc: 
description: Synchronize the output channels. 
method: get
name: sync
parameters:
- default: null
  description: Mask value
  name: mask
  type: integer
  unit: null
summary: sync
---

<headers/>
<parameters/>

<code-group>
<code-block title="Python">
```python
from moku.instruments import MultiInstrument, CloudCompile
m = MultiInstrument('192.168.###.###', platform_id=2)
cc = m.set_instrument(1, CloudCompile)
# set instrument in slot 2
# set connections
cc.sync(1)
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuMultiInstrument('192.168.###.###', 'platform_id', 2);
cc = m.set_instrument(1, MokuCloudCompile);
% set instrument in slot 2
% set connections
cc.sync(1);
```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"mask": 1}'\
        http://<ip>/api/slot1/cloudcompile/sync
```
</code-block>


</code-group>
