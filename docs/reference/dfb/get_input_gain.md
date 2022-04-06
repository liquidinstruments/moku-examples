---
additional_doc: null
description: Get the pre-filter gain for a given channel
method: get
name: get_input_gain
parameters:
- default: null
  description: Target channel
  name: channel
  param_range:
   mokugo: 1, 2
   mokupro: 1, 2, 3, 4
  type: integer
  unit: null
summary: get_input_gain
---


<headers/>


<parameters/>

### Examples

<code-group>
<code-block title="Python">
```python
from moku.instruments import DigitalFilterBox
i = DigitalFilterBox('192.168.###.###')
i.get_input_gain(1)
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuDigitalFilterBox('192.168.###.###');
m.get_input_gain(1);
```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"channel":1}'\
        http://<ip>/api/digitalfilterbox/get_input_gain
```
</code-block>

</code-group>

### Sample response
```json
{"gain":10.0}
```