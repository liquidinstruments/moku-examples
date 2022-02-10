---
additional_doc: The PLL in the LIA instrument is driven by the Moku's Input 2 and can optionally be used as a demodulation source. See `set_demodulation`.
description: Sets the tracking bandwidth of the PLL.
method: post
name: enable_rollmode
parameters:
- default: null
  description: PLL Bandwidth
  name: bandwidth
  param_range: 10kHz, 2k5Hz, 600Hz, 150Hz, 40Hz, 10Hz
  type: string
  unit: null
- default: true
  description: Disable all implicit conversions and coercions.
  name: strict
  param_range: null
  type: boolean
  unit: null
summary: enable_rollmode
group: Input PLL
---

<headers/>
<parameters/>
