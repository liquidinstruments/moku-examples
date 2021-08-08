---
additional_doc: null
description: Set the left- and right-hand span for the time axis. Units are seconds
  relative to the trigger point.
method: post
name: set_timebase
parameters:
- default: null
  description: Time from the trigger point to the left of screen. This may be negative
    (trigger on-screen) or positive (trigger off the left of screen).
  name: t1
  param_range: null
  type: number
  unit: Seconds
- default: null
  description: Time from the trigger point to the right of screen. (Must be a positive
    number, i.e. after the trigger event)
  name: t2
  param_range: null
  type: number
  unit: Seconds
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

Usage in clients, 

<code-group>
<code-block title="Python">
```python
from moku.instruments import Oscilloscope
# Configure the instrument
# View +- 1 ms i.e. trigger in the centre
i.set_timebase(-1e-3, 1e-3)
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuOscilloscope('192.168.###.###', true);
%% Configure the instrument
% View +- 1 ms i.e. trigger in the centre
m.set_timebase(-1e-3, 1e-3);
```
</code-block>
</code-group>