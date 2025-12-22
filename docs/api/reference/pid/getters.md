---
additional_doc: Along with regular configuration methods,  PID Controller supports following getter functions.
name: getters
description: PID Controller - getter functions
getters: 
- summary: get_control_matrix
  description: Get the linear combination of ADC input signals for a given PID channel.
  parameters:
  - default: null
    description: Target channel
    name: channel
    param_range:
     mokugo: 1, 2
     mokulab: 1, 2
     mokupro: 1, 2, 3, 4
     mokudelta: 1, 2, 3, 4
    type: integer
    unit: null
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
     mokudelta: 1, 2, 3, 4
    type: integer
    unit: null
- summary: get_input_offset
  description: Get the input offset for given input channel
  parameters:
  - default: null
    description: Target channel
    name: channel
    param_range:
     mokugo: 1, 2
     mokulab: 1, 2
     mokupro: 1, 2, 3, 4
     mokudelta: 1, 2, 3, 4
    type: integer
    unit: null
- summary: get_output_offset
  description: Get the output offset for given input channel
  parameters:
  - default: null
    description: Target channel
    name: channel
    param_range:
     mokugo: 1, 2
     mokulab: 1, 2
     mokupro: 1, 2, 3, 4
     mokudelta: 1, 2, 3, 4
    type: integer
    unit: null
- summary: get_output_gain
  description: Get the output gain for given input channel
  parameters:
  - default: null
    description: Target channel
    name: channel
    param_range:
     mokugo: 1, 2
     mokulab: 1, 2
     mokupro: 1, 2, 3, 4
     mokudelta: 1, 2, 3, 4
    type: integer
    unit: null
- summary: get_output_limit
  description: Get the output limit for given input channel
  parameters:
  - default: null
    description: Target channel
    name: channel
    param_range:
     mokugo: 1, 2
     mokulab: 1, 2
     mokupro: 1, 2, 3, 4
     mokudelta: 1, 2, 3, 4
    type: integer
    unit: null
---
<headers/>
<getters/>
