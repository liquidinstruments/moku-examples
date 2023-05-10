---
additional_doc: Along with regular configuration methods, Spectrum Analyzer supports following getter functions.
description: Getters
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
- summary: get_rbw
  description: Returns the current resolution bandwidth (Hz)
- summary: get_span
  description: Get the current frequency span, frequency1 and frequency2
- summary: get_window
  description: Get the currently configured window function
---
<headers/>
<getters/>
