---
additional_doc: null
description: Get the auto acquired frequency of a channel
method: get
name: get_auto_acquired_frequency
summary: get_auto_acquired_frequency
available_on: "mokupro"

parameters:
- default: null
  description: Target channel
  name: channel
  param_range: 1, 2, 3, 4
  type: integer
  unit: null
- default: true
  description: Disable all implicit conversions and coercions.
  name: strict
  param_range: null
  type: boolean
  unit: null

---





<headers/>
<parameters/>


### Examples

<code-group>
<code-block title="Python">
```python
from moku.instruments import MokuPhasemeter
i = MokuPhasemeter('192.168.###.###', force_connect=False)
i.get_auto_acquired_frequency(channel=1)
```
</code-block>

<code-block title="MATLAB">
```matlab
i = MokuPhasemeter('192.168.###.###', true);
i.get_auto_acquired_frequency(1);
```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        http://<ip>/api/phasemeter/get_auto_acquired_frequency
```
</code-block>

</code-group>