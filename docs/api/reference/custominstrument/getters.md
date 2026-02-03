---
additional_doc: Along with regular configuration methods, Custom Instrument supports following getter functions.
name: getters
description: Custom Instrument - getter functions
getters:
- summary: get_controls
  description: Get all the control register values
- summary: get_control
  description: Get the value for a given control register index
  parameters:
  - default: null
    description: Target control register
    name: idx
    type: integer
    allowed_values: 1 to 16
    unit: null
- summary: get_interpolation
  description: Get the output interpolation flag for a given channel
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
- summary: get_status
  description: Get all the status register values
---
<headers/>
<getters/>
