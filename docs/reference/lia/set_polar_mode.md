---
additional_doc: null
description: Sets the range of the signal input to the Polar conversion block.
method: post
name: set_polar_mode
parameters:
- default: null
  description: Gain range
  name: range
  param_range: 2Vpp, 7.5mVpp, 25uVpp
  type: string
  unit: null
- default: true
  description: Disable all implicit conversions and coercions.
  name: strict
  param_range: null
  type: boolean
  unit: null
summary: set_polar_mode
---

<headers/>

:::tip Enabling the Converter
To enable the polar conveter, the R/Theta signals must also be selected for output from the instrument, see [set_outputs](./set_outputs.md). You should set the smallest range that accommodates your demodulated signal without saturating.
:::

<parameters/>
