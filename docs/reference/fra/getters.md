---
additional_doc: Along with regular configuration methods,  Frequency Response Analyzer supports following getter functions.
name: getters
description: Frequency Response Analyzer  - getter functions
getters: 
- summary: get_harmonic_multiplier
  description: Get the current harmonic multiplier
- summary: get_output
  description: Get output amplitude and offset for the given channel
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
  description: Get output load
  parameters:
  - default: null
    description: Target channel
    name: channel
    param_range:
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
- summary: get_sweep
  description: Get the current sweep configuration
  additional_doc: Response includes start_frequency, stop_frequency, averaging_cycles, settling_cycles, averaging_time, settling_time, num_points, linear_scale, estimated_sweep_time
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

---
<headers/>
<getters/>
