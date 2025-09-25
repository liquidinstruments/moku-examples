---
title: disable_output
additional_doc: null
description: Turn off the output channels
method: post
name: disable_output
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
summary: disable_output
---

<headers/>
<parameters/>

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import Phasemeter

i = Phasemeter('192.168.###.###', force_connect=True)
# Disable Out 1
i.disable_output(channel=1)

```

</code-block>

<code-block title="MATLAB">

```matlab
i = MokuPhasemeter('192.168.###.###', force_connect=true);
% Disable Out 1
i.disable_output(1);

```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"channel": 1}'\
        http://<ip>/api/phasemeter/disable_output
```

</code-block>

</code-group>
