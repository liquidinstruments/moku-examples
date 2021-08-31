---
additional_doc: null
description: Sets the left- and right-hand span for the time axis. Units are seconds
  relative to the trigger point.
method: post
name: set_timebase
parameters:
- default: null
  description: Time from the trigger point to the left of screen.
  name: t1
  param_range: null
  type: number
  unit: null
- default: null
  description: Time from the trigger point to the right of screen. (Must be a positive
    number, i.e. post trigger event)
  name: t2
  param_range: null
  type: number
  unit: null
- default: null
  description: Toggle Roll Mode
  name: roll_mode
  param_range: null
  type: boolean
  unit: null
- default: true
  description: Disable all implicit conversions and coercions.
  name: strict
  param_range: null
  type: boolean
  unit: null
summary: set_timebase
available_on: "mokugo"
---





<headers/>
<parameters/>