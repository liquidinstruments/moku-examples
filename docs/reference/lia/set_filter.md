---
additional_doc: null
description: Configures the LIA low-pass filter
method: post
name: set_filter
parameters:
- default: null
  description: Filter corner frequency
  name: corner_frequency
  type: number
  unit: Hz
- default: Slope6dB
  description: Filter slope per octave
  name: slope
  param_range: Slope6dB, Slope12dB, Slope18dB, Slope24dB
  type: string
  unit: null
- default: true
  description: Disable all implicit conversions and coercions.
  name: strict
  param_range: null
  type: boolean
  unit: null
summary: set_filter
---

<headers/>
<parameters/>
