---
additional_doc: null
description: Set input signal offset
method: post
name: set_input_offset
parameters:
- default: null
  description: Target channel
  name: channel
  param_range:
   mokugo: 1, 2
   mokupro: 1, 2, 3, 4
  type: integer
  unit: null
- default: null
  description: Output DC offset
  name: offset
  param_range: -5 to 5
  type: number
  unit: V
- default: true
  description: Disable all implicit conversions and coercions.
  name: strict
  param_range: null
  type: boolean
  unit: null
summary: set_input_offset
---





<headers/>
<parameters/>

### Examples

<code-group>
<code-block title="Python">
```python
from moku.instruments import PIDController
i = PIDController('192.168.###.###', force_connect=False)
# Configure instrument to desired state
# Set input offset to 5VDC
i.set_input_offset(1, offset=5)
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuPIDController('192.168.###.###', true);
% Configure instrument to desired state
% Set input offset to 5VDC
m.set_input_offset(1, 'offset', 5);
```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"channel": 1, "offset": 5}'\
        http://<ip>/api/firfilter/set_input_offset
```
</code-block>

</code-group>