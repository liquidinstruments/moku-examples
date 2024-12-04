---
description:
    Set the left- and right-hand span for the time axis. Units are seconds
    relative to the trigger point.
method: post
name: set_timebase
parameters:
    - default: null
      description:
          Time from the trigger point to the left of screen. This may be negative
          (trigger on-screen) or positive (trigger off the left of screen).
      name: t1
      param_range: null
      type: number
      unit: Seconds
    - default: null
      description:
          Time from the trigger point to the right of screen. (Must be a positive
          number, i.e. after the trigger event)
      name: t2
      param_range: null
      type: number
      unit: Seconds
    - name: max_length
      description:
          Requested maximum frame length. The subsequent calls to `get_data` will
          return frames with as close to this many points as possible without going
          over. Achievable frame lengths depend on the specific timebase and hardware
          version, so the user code should be written to work with frame lengths
          that change when the timebase changes.
      type: number
      default: 1024
      param_range: 1024, 2048, 4096, 8192
      unit: points
    - default: true
      description: Disable all implicit conversions and coercions.
      name: strict
      param_range: null
      type: boolean
      unit: null
summary: set_timebase
---

<headers/>
<parameters/>

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import Oscilloscope
i = Oscilloscope('192.168.###.###')
# Configure the instrument
# View +- 1 ms i.e. trigger in the centre
i.set_timebase(-1e-3, 1e-3)
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuOscilloscope('192.168.###.###');
%% Configure the instrument
% View +- 1 ms i.e. trigger in the centre
m.set_timebase(-1e-3, 1e-3);
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"t1": -1e-3, "t2": 1e-3}'\
        http://<ip>/api/oscilloscope/set_timebase
```

</code-block>

</code-group>

### Sample response

```json
{
    "offset": 0.0,
    "span": 0.002
}
```
