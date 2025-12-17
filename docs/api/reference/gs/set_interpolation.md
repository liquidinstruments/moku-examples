---
additional_doc: null
description: Configure interpolation mode for outgoing waveform data
method: post
name: set_interpolation
parameters:
    - default: null
      description: Interpolation mode
      name: mode
      param_range: None, Linear
      type: string
      unit: null
    - default: true
      description: Disable all implicit conversions and coercions.
      name: strict
      param_range: null
      type: boolean
      unit: null
summary: set_interpolation
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
gs.set_interpolation(mode='Linear', strict=True)
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuMultiInstrument('192.168.###.###', 3)
gs = m.set_instrument(1, @MokuGigabitStreamer)
gs.set_interpolation('Linear', "strict": true)
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
       -H 'Content-Type: application/json'\
       --data '{"mode": "Linear", "strict": true}'\
       http://<ip>/<slot>/api/gs/set_interpolation
```

</code-block>

</code-group>

### Sample response

```json
{
    "interpolation_mode": "Linear"
}
```
