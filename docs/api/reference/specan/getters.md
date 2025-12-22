---
additional_doc: Along with regular configuration methods, Spectrum Analyzer supports following getter functions.
name: getters
description: Spectrum Analyzer - getter functions
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
- summary: get_rbw
  description: Returns the current resolution bandwidth (Hz)
- summary: get_span
  description: Get the current frequency span, frequency1 and frequency2
- summary: get_window
  description: Get the currently configured window function
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
