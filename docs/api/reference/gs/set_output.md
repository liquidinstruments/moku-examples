---
additional_doc: null
description: Enable and configure an output channel
method: post
name: set_output
parameters:
    - default: null
      description: Target channel
      name: channel
      param_range: 1, 2
      type: integer
      unit: null
    - default: null
      description: Enable output signal
      name: enable
      param_range: null
      type: boolean
      unit: null
    - default: null
      description: Gain in dB
      name: gain
      param_range: null
      type: number
      unit: null
    - default: null
      description: Offset in V
      name: offset
      param_range: null
      type: number
      unit: null
    - default: true
      description: Disable all implicit conversions and coercions.
      name: strict
      param_range: null
      type: boolean
      unit: null
summary: set_output
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
gs.set_output(channel=1, enable=True, gain=0.0, offset=0.0, strict=True)
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuMultiInstrument('192.168.###.###', 3)
gs = m.set_instrument(1, @MokuGigabitStreamer)
gs.set_output(1, true, 0.0, 0.0, 'strict', true)
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
       -H 'Content-Type: application/json'\
       --data '{"channel": 1,"enable": true, "gain": 0.0, "offset: 0.0, "strict": true}'\
       http://<ip>/<slot>/api/gs/set_output
```

</code-block>

</code-group>

### Sample response

```json
{
    "enabled": "1",
    "gain": "0",
    "offset": "0",
}
```
