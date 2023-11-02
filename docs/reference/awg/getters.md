---
additional_doc: Along with regular configuration methods,  Arbitrary Waveform Generator supports following getter functions.
name: getters
description: Arbitrary Waveform Generator - getter functions 
getters: 
- summary: get_frontend
  description: Get the input impedance, coupling, and range for given input channel
  parameters:
  - default: null
    description: Target channel
    name: channel
    param_range:
     mokugo: 1, 2
     mokulab: 1, 2
     mokupro: 1, 2, 3, 4
    type: integer
    unit: null
- summary: get_output_load
  description: Get the output load for a given output channel
  parameters:
  - default: null
    description: Target channel
    name: channel
    param_range:
     mokugo: 1, 2
     mokulab: 1, 2
     mokupro: 1, 2, 3, 4
    type: integer
    unit: null
  deprecated: true
  deprecated_text: get_output_load is deprecated, use get_output_termination to get the output termination
- summary: get_output_termination
  description: Get output termination
  parameters:
  - default: null
    description: Target channel
    name: channel
    param_range:
     mokulab: 1, 2
     mokupro: 1, 2, 3, 4
    type: integer
    unit: null
---
<headers/>
<getters/>
