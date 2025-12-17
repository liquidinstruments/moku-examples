---
additional_doc: null
description: Configure acquisition mode and sample rate
method: post
name: set_acquisition
parameters:
    - default: null
      description: Acquisition mode
      name: mode
      param_range: Normal, Precision
      type: string
      unit: null
    - default: null
      description: Target samples per second
      name: sample_rate
      param_range:
        gigabitstreamer: 5000 to 312500000
        gigabitstreamerplus: 5000 to 5000000000
      type: number
      unit: null
    - default: true
      description: Disable all implicit conversions and coercions.
      name: strict
      param_range: null
      type: boolean
      unit: null
summary: set_acquisition
available_on: 'Moku:Delta'
---

<headers/>

::: tip sample_rate
The sampling rate is achieved through decimation, where the decimation factor must be a factor of 2 or a multiple of 16. Your sampling rate may be coerced to the nearest allowed value, if a sampling rate does not meet these criteria.
Sampling rate has a maximum rate that is split across the enabled input channels. E.g. if the maximum sampling rate for your configuration is 312.5 MHz with one input channel enabled, the maximum sampling rate is effectively halved (156.25 MHz) if a second input channel is enabled.
:::
<parameters/>

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import MultiInstrument, GigabitStreamer
m = MultiInstrument('192.168.###.###', platform_id=3)
gs = m.set_instrument(1, GigabitStreamer)
gs.set_acquisition(mode='Normal', sample_rate=5e3, strict=True)
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuMultiInstrument('192.168.###.###', 3)
gs = m.set_instrument(1, @MokuGigabitStreamer)
gs.set_acquisition('Normal', 5e3, 'strict', true)
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
       -H 'Content-Type: application/json'\
       --data '{"mode": "Normal","sample_rate": 5e3, "strict": True}'\
       http://<ip>/<slot>/api/gs/set_acquisition
```

</code-block>

</code-group>

### Sample response

```json
{
    "mode": "Normal",
    "sample_rate": "5000",
}
```
