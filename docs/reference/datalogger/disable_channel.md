---
additional_doc: null
description: Disable an input channel
method: post
name: disable_channel
parameters:
- default: null
  description: Target channel
  name: channel
  param_range:
   mokugo: 1, 2
   mokulab: 1, 2
   mokupro: 1, 2, 3, 4
  type: integer
  unit: null
- default: true
  description: Flag to enable or disable channel.
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
summary: disable_output
---





<headers/>
<parameters/>

<code-group>
<code-block title="Python">
```python
from moku.instruments import Datalogger

i = Datalogger('192.168.###.###')
# Disable channel 1
i.disable_channel(1)
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuDatalogger('192.168.###.###');
% Disable channel 1
m.disable_channel(1);
```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"channel": 1}'\
        http://<ip>/api/datalogger/disable_channel
```
</code-block>

</code-group>
