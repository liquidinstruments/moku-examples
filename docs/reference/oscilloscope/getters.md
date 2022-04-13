---
additional_doc: Along with regular configuration methods, Oscilloscope supports following getter functions.
name: getters
description: Oscilloscope - getter functions
getters: 
- summary: get_acquisition_mode
  description: Get the current acquisition mode
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
- summary: get_interpolation
  description: Get the current interpolation
- summary: get_samplerate
  description: Get the current sample rate
- summary: get_sources
  description: Get the status and source for every available channel

---
<headers/>
<getters/>
