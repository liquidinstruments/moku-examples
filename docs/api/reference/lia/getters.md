---
additional_doc: Along with regular configuration methods, Lockin Amplifier supports following getter functions.
name: getters
description: Lock-in Amplifier  - getter functions
getters:
    - summary: get_aux_output
      description: Get the current amplitude and frequency for auxiliary output
    - summary: get_pll
      description: Get the current PLL configuration
    - summary: get_filter
      description: Get the current filter parameters corner and slope
    - summary: get_gain
      description: Get the current main and aux gains
    - summary: get_demodulation
      description: Get the current demodulator configuration
    - summary: get_outputs
      description: Get the current main and aux output configuration
    - summary: get_polar_theta_range
      description: If polar mode is enabled,returns the current range
    - summary: get_frontend
      description: Get the input impedance, coupling, and attenuation for given input channel
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
---

<headers/>
<getters/>
