---
additional_doc: null
description: Sends the signal from the first input to all phasemeter channels
method: post
name: enable_single_input
parameters:
    - default: true
      description: Enables single input
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
summary: enable_single_input
---

<headers/>
<parameters/>

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import Phasemeter

i = Phasemeter('192.168.###.###', force_connect=True)
i.enable_single_input()
```

</code-block>

<code-block title="MATLAB">

```matlab
i = MokuPhasemeter('192.168.###.###', force_connect=true);
i.enable_single_input();
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"disable": true}'\
        http://<ip>/api/phasemeter/enable_single_input
```

</code-block>

</code-group>
