---
additional_doc: null
description: Get the linear combination of ADC input signals for a given DFB channel.
method: get
name: get_control_matrix
parameters:
- default: null
  description: Target channel
  name: channel
  param_range:
   mokugo: 1, 2
   mokupro: 1, 2, 3, 4
  type: integer
  unit: null
summary: get_control_matrix
---

<headers/>
<parameters/>


### Examples

<code-group>
<code-block title="Python">
```python
from moku.instruments import DigitalFilterBox
i = DigitalFilterBox('192.168.###.###')
# Configure the input control matrix
i.get_control_matrix(2)
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuDigitalFilterBox('192.168.###.###');
% Configure the input control matrix
m.get_control_matrix(2);
```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"channel": 2}'\
        http://<ip>/api/digitalfilterbox/get_control_matrix
```
</code-block>

</code-group>

### Sample response
```json
{
  "input_gain1": 1.0,
  "input_gain2": 0.0
}
```