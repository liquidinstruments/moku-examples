---
additional_doc: null
description: Sets output gain levels
method: post
name: set_gain
parameters:
- default: null
  description: Main output gain
  name: main
  type: number
  unit: dB
- default: null
  description: Auxilliary output gain
  name: aux
  type: number
  unit: dB
- default: true
  description: Disable all implicit conversions and coercions.
  name: strict
  param_range: null
  type: boolean
  unit: null
summary: set_gain
---

<headers/>
<parameters/>

:::tip Note
Output inversion is currently not supported by the API
:::
