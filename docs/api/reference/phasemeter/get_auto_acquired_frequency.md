---
additional_doc: null
description: Get the auto acquired frequency of a channel
method: get
name: get_auto_acquired_frequency
summary: get_auto_acquired_frequency

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
from moku.instruments import Phasemeter
i = Phasemeter('192.168.###.###', force_connect=True)
i.get_auto_acquired_frequency(channel=1)
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuPhasemeter('192.168.###.###', force_connect=true);
m.get_auto_acquired_frequency(1);
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        --data '{"channel":1}'\
        http://<ip>/api/phasemeter/get_auto_acquired_frequency
```

</code-block>

</code-group>

### Sample response

```json
30000000
```
