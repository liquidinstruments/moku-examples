---
additional_doc: null
description: Configures the demodulation source and optionally its frequency and phase
method: post
name: set_demodulation
parameters:
- default: null
  description: The demodulation source
  name: mode
  param_range: Internal, External, ExternalPLL, None
  type: string
  unit: null
- default: 1000000
  description: Frequency of internally-generated demod source
  name: frequency
  type: number
  unit: Hz
- default: 0
  description: Phase of internally-generated demod source
  name: phase
  type: number
  unit: degrees
- default: true
  description: Disable all implicit conversions and coercions.
  name: strict
  param_range: null
  type: boolean
  unit: null
summary: set_demodulation
---

<headers/>
<parameters/>
