---
additional_doc: In the PID instrument, this timebase refers to the Monitor unit. See `set_monitor`.
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
      unit: null
    - default: null
      description:
          Time from the trigger point to the right of screen. (Must be a positive
          number, i.e. after the trigger event)
      name: t2
      param_range: null
      type: number
      unit: null
    - default: true
      description: Disable all implicit conversions and coercions.
      name: strict
      param_range: null
      type: boolean
      unit: null
summary: set_timebase
group: Monitors
---

<headers/>
<parameters/>

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import FIRFilterBox
i = FIRFilterBox('192.168.###.###')
# Set instrument to desired state
# View +- 1 ms i.e. trigger in the centre
i.set_timebase(-1e-3, 1e-3)
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuFIRFilterBox('192.168.###.###');
% Set instrument to desired state
% View +- 1 ms i.e. trigger in the centre
m.set_timebase(-1e-3, 1e-3);
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"t1": -1e-3, "t2": 1e-3}'\
        http://<ip>/api/firfilter/set_timebase
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
