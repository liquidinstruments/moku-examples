---
additional_doc: null
description: Set input signal gain
method: post
name: set_input_gain
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
- default: null
  description: Input gain
  name: gain
  param_range: -5 to 5
  type: number
  unit: dB
- default: true
  description: Disable all implicit conversions and coercions.
  name: strict
  param_range: null
  type: boolean
  unit: null
summary: set_input_gain
---





<headers/>
<parameters/>

### Examples

<code-group>
<code-block title="Python">
```python
from moku.instruments import FIRFilterBox
i = FIRFilterBox('192.168.###.###')
# Configure instrument to desired state
# Set input gain to 5dB
i.set_input_gain(1, gain=5)
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuFIRFilterBox('192.168.###.###');
% Configure instrument to desired state
% Set input gain to 5dB
m.set_input_offset(1, 'gain', 5);
```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"channel": 1, "gain": 5}'\
        http://<ip>/api/firfilter/set_input_gain
```
</code-block>

</code-group>

### Sample response,
```json
{
  "gain": 5.0
}
```