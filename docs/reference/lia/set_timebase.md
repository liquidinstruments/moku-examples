---
additional_doc: This applies to the embedded monitor in the LIA. The monitor inputs can be configured with `set_monitor`.
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
group: Monitors
---

<headers/>
<parameters/>
