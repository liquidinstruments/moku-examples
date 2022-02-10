---
additional_doc: In addition to configuring this Auxilliary sine wave output, it must be routed to an
    actual output channel using `set_outputs`
description: Configures the Auxilliary sine wave generator.
method: post
name: set_aux_output
parameters:
- default: null
  description: Sine wave frequency
  name: frequency
  type: number
  unit: Hz
- default: null
  description: Sine wave amplitude
  name: amplitude
  type: number
  unit: V
- default: true
  description: Disable all implicit conversions and coercions.
  name: strict
  param_range: null
  type: boolean
  unit: null
summary: set_aux_output
---

<headers/>
<parameters/>
