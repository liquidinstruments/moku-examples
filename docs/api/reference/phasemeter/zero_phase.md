---
additional_doc: null
description: Zeroes the phase on given Phasemeter channel.
method: post
name: zero_phase
parameters:
    - default: null
      description: Target channel
      name: channel
      param_range:
          mokugo: 1, 2
          mokulab: 1, 2
          mokupro: 1, 2, 3, 4
          mokudelta: 1, 2, 3, 4, 5, 6, 7, 8
      type: integer
      unit: null
summary: zero_phase
---

<headers/>
<parameters/>

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import Phasemeter
i = Phasemeter('192.168.###.###', force_connect=True)
# Sets Input 1 phase to zero
i.zero_phase(1, True)
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuPhasemeter('192.168.###.###', force_connect=true);
% Sets Input 1 phase to zero
m.zero_phase(1, true)
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"channel": 2, "enabled": true}'\
        http://<ip>/api/phasemeter/zero_phase
```

</code-block>

</code-group>
