---
additional_doc: null
description: Toggle the output interpolation for a given channel.
method: post
name: set_interpolation
parameters:
- default: null
  description: Target channel
  name: channel
  param_range: 
    mokugo: 1, 2
    mokupro: 1, 2, 3, 4
  type: integer
  unit: null
summary: set_interpolation
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
cc.set_interpolation(1, True)
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuMultiInstrument('192.168.###.###', 'platform_id', 2);
cc = m.set_instrument(1, MokuCloudCompile);
% set instrument in slot 2
% set connections
cc.set_interpolation(1, true);
```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"channel": 1, "enable": true}'\
        http://<ip>/api/slot1/cloudcompile/set_interpolation
```
</code-block>


</code-group>