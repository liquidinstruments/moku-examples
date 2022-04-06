---
additional_doc: Along with regular configuration methods,  Arbitrary Waveform Generator supports following getter functions.
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
     mokupro: 1, 2, 3, 4
    type: integer
    unit: null

---
<headers/>
<getters/>
