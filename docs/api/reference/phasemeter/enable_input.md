---
additional_doc: null
description: Enable or disable the Phasemeter channel input(s).
method: post
name: enable_input
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
    - default: True
      description: Enable/Disable input signal
      name: enable
      param_range: null
      type: boolean
      unit: null
    - default: true
      description: Disable all implicit conversions and coercions.
      name: strict
      param_range: null
      type: boolean
      unit: null
summary: enable_input
---

<headers/>
<parameters/>

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import Phasemeter
i = Phasemeter('192.168.###.###')
# Enable Input Signal
i.enable_input(1, True)
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuPhasemeter('192.168.###.###');
% Enable Input Signal
m.enable_input(1, true)
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"channel": 2, "enabled": true}'\
        http://<ip>/api/phasemeter/enable_input
```

</code-block>

</code-group>
