---
additional_doc: Along with regular configuration methods, Logic Analyzer supports following getter functions.
name: getters
description: Logic Analyzer -  getter functions
getters: 
- summary: get_decoder
  description: Gets the decoder configuration for the given ID.
  parameters:
  - default: null
    description: Target channel
    name: channel
    param_range: 1, 2
    type: integer
    unit: null
- summary: get_pattern_generator
  description: Gets the configuration for a given pattern generator ID.
  parameters:
  - default: null
    description: Target channel
    name: channel
    param_range: 1, 2
    type: integer
    unit: null
- summary: get_source
  description: Gets the type of Logic Analyzer's input source.
- summary: get_analog_mode
  description: Gets the threshold voltages of converting analog inputs to digital.
- summary: get_pins
  description: Gets the the states of Logic Analyzer's digital pins.
  deprecated: true
  deprecated_text: get_pins is deprecated, use get_pin_mode to get the state of digital pins
- summary: get_pin_mode
  description: Gets the state of Logic Analyzer's digital pin.
  parameters:
  - default: null
    description: Target pin
    name: channel
    param_range:
     mokugo: 1 to 16
    type: integer
    unit: null
- summary: get_pin
  description: Gets the the states of Logic Analyzer's digital pins.
  deprecated: true
  deprecated_text: get_pin is deprecated, use get_pin_mode to get the state of digital pins
  parameters:
  - default: null
    description: Target pin
    name: channel
    param_range:
     mokugo: 1 to 16
    type: integer
    unit: null

---
<headers/>
<getters/>
