---
additional_doc: Along with regular configuration methods, Multi-instrument supports following getter functions.
name: getters
description: Multi-instrument -  getter functions
getters: 
- summary: get_connections
  description: Get configured connections between all the available slots.
- summary: get_dio
  description: Get the current DIO direction configuration.
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
- summary: get_instruments
  description: Get configured instruments for all available slots.
- summary: get_output
  description: Get the current output relay configuration
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
