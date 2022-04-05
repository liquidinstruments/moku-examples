---
additional_doc: This mode is only available for relatively long timebases. The exact range is hardware-dependent
    but should typically be used at or above 1 second. In Roll Mode, trigger settings are ignored but not changed.
description: Enables and disables Roll Mode X-axis behaviour
method: post
name: enable_rollmode
parameters:
- default: null
  description: Whether Roll Mode should be used
  name: roll
  param_range: true, false
  type: boolean
  unit: null
- default: true
  description: Disable all implicit conversions and coercions.
  name: strict
  param_range: null
  type: boolean
  unit: null
summary: enable_rollmode
group: Monitors
---

<headers/>
<parameters/>
