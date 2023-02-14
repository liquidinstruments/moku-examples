---
additional_doc: Along with regular configuration methods,  Datalogger supports following getter functions.
name: getters
description: Datalogger - getter functions 
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
- summary: get_samplerate
  description: Get the current rate at which samples are acquired
- summary: get_stream_status
  description: Get the streaming session status, including memory usage, state and error information
- summary: get_output_load
  description: Get the output load for a given output channel
  parameters:
  - default: null
    description: Target channel
    name: channel
    param_range: 1, 2, 3, 4
    type: integer
    unit: null

---
<headers/>
<getters/>
