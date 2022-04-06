---
additional_doc: null
description: Get the input impedance, coupling, and range for given channel.
method: get
name: get_frontend
parameters:
- default: null
  description: Target channel
  name: channel
  param_range:
   mokugo: 1, 2
   mokupro: 1, 2, 3, 4
  type: integer
  unit: null
summary: get_frontend
---


<headers/>


<parameters/>

### Examples

<code-group>
<code-block title="Python">
```python
from moku.instruments import DigitalFilterBox
i = DigitalFilterBox('192.168.###.###')
i.get_frontend(1)
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuDigitalFilterBox('192.168.###.###');
m.get_frontend(1);
```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"channel":1}'\
        http://<ip>/api/digitalfilterbox/get_frontend
```
</code-block>

</code-group>

### Sample response
```json
{
  "coupling":"DC",
  "impedance":"1MOhm",
  "range":"10Vpp"
}
```