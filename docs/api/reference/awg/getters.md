---
additional_doc: Along with regular configuration methods,  Arbitrary Waveform Generator supports following getter functions.
name: getters
description: Arbitrary Waveform Generator - getter functions 
getters: 
- summary: get_frontend
  description: Get the input impedance, coupling, bandwidth, and range for given input channel
  parameters:
  - default: null
    description: Target channel
    name: channel
    param_range:
     mokugo: 1, 2
     mokulab: 1, 2
     mokupro: 1, 2, 3, 4
     mokudelta: 1, 2, 3, 4, 5, 6, 7, 8
    type: integer
    unit: null
- summary: get_output_load
  deprecated: true
  deprecated_msg: This method is deprecated and will be removed soon. Use **get_output_termination** instead.
  description: Get the output load for a given output channel
  parameters:
  - default: null
    description: Target channel
    name: channel
    param_range:
     mokugo: 1, 2
     mokulab: 1, 2
     mokupro: 1, 2, 3, 4
     mokudelta: 1, 2, 3, 4, 5, 6, 7, 8
    type: integer
    unit: null
- summary: get_output_termination
  description: Get the output termination for a given output channel
  parameters:
  - default: null
    description: Target channel
    name: channel
    param_range:
     mokulab: 1, 2
     mokupro: 1, 2, 3, 4
     mokudelta: 1, 2, 3, 4, 5, 6, 7, 8
    type: integer
    unit: null
---
<headers/>
<getters/>
