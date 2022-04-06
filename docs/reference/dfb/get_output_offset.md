---
additional_doc: null
description: Get the ost-filter offset for a given channel
method: get
name: get_output_offset
parameters:
- default: null
  description: Target channel
  name: channel
  param_range:
   mokugo: 1, 2
   mokupro: 1, 2, 3, 4
  type: integer
  unit: null
summary: get_output_offset
---


<headers/>


<parameters/>

### Examples

<code-group>
<code-block title="Python">
```python
from moku.instruments import DigitalFilterBox
i = DigitalFilterBox('192.168.###.###')
i.get_output_offset(1)
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuDigitalFilterBox('192.168.###.###');
m.get_output_offset(1);
```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"channel":1}'\
        http://<ip>/api/digitalfilterbox/get_output_offset
```
</code-block>

</code-group>

### Sample response
```json
{"offset":2.0}
```