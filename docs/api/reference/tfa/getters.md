---
# get_acquisition_mode.md
# get_event_detector.md
# get_frontend.md
# get_interpolation.md
# get_interval_analyzer.md
# get_register.md
additional_doc: Along with regular configuration methods, Time & Frequency Analyzer supports following getter functions.
name: getters
description: Time & Frequency Analyzer - getter functions
getters:
- summary: get_acquisition_mode
  description: Returns the acquisition mode of Time & Frequency Analyzer 
- summary: get_event_detector
  description: Get the event-detector settings for the given numerical id
  parameters:
  - default: null
    description: Numerical id of the event
    name: id
    param_range:
     mokugo: 1, 2
     mokulab: 1, 2
     mokupro: 1, 2, 3, 4
    type: integer
    unit: null
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
- summary: get_interpolation
  description: Get the currently configured interpolation
- summary: get_interval_analyzer
  description: Get the interval configuration for the given numerical id
  parameters:
  - default: null
    description: Numerical id of the interval
    name: id
    param_range:
     mokulab: 1, 2
     mokupro: 1, 2, 3, 4
    type: integer
    unit: null
---
<headers/>
<getters/>
