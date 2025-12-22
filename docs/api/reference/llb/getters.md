---
additional_doc: Along with regular configuration methods,  Laser Lock Box supports following getter functions.
summary: Laser Lock Box - getter functions
name: getters

getters: 
- summary: get_aux_oscillator
  description: Get the current state of auxiliary oscillator
- summary: get_demodulation
  description: Get the current demodulation configuration
- summary: get_frontend
  description: Get the input impedance, coupling, bandwidth, and range for given input channel
  parameters:
  - default: null
    description: Target channel
    name: channel
    param_range: 1, 2
    type: integer
    unit: null
- summary: get_scan_oscillator
  description: Get the current state of scan oscillator
- summary: get_output_limit
  description: Get the limiter voltage for given output channel
  parameters:
  - default: null
    description: Target channel
    name: channel
    param_range: 1, 2
    type: integer
    unit: null
- summary: get_output_offset
  description: Get the output offset for given input channel
  parameters:
  - default: null
    description: Target channel
    name: channel
    param_range:
     mokulab: 1, 2
     mokugo: 1, 2
     mokupro: 1, 2, 3, 4
     mokudelta: 1, 2, 3, 4
    type: integer
    unit: null
- summary: get_pll
  description: Get the current PLL state
- summary: get_setpoint
  description: Get the current setpoint voltage
---
<headers/>
<getters/>
