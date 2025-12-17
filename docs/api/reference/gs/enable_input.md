---
additional_doc: null
description: Enable or disable an input channel
method: post
name: enable_input
parameters:
    - default: null
      description: Target channel
      name: channel
      param_range:
        mokudelta: 1, 2, 3, 4
      type: integer
      unit: null
    - default: true
      description: Enable input signal
      name: enable
      param_range: null
      type: boolean
      unit: null
    - default: true
      description: Disable all implicit conversions and coercions
      name: strict
      param_range: null
      type: boolean
      unit: null
summary: enable_input
available_on: 'Moku:Delta'
---

<headers/>
<parameters/>

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import MultiInstrument, GigabitStreamer
m = MultiInstrument('192.168.###.###', platform_id=3)
gs = m.set_instrument(1, GigabitStreamer)
gs.enable_input(channel=1, enable=True, strict=True)
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuMultiInstrument('192.168.###.###', 3)
gs = m.set_instrument(1, @MokuGigabitStreamer)
gs.enable_input(1, true, 'strict', true)
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
       -H 'Content-Type: application/json'\
       --data '{"channel": 1,"enable": true, "strict": true}'\
       http://<ip>/<slot>/api/gs/enable_input
```

</code-block>

</code-group>

### Sample response
<!-- When readback works response should look like

```json
{
    "strict": "True",
    "channel": "1",
    "enable": "True"
}
``` -->
