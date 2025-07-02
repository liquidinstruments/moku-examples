---
additional_doc: Along with regular configuration methods, Phasemeter supports following getter functions.
name: getters
description: Phasemeter - getter functions
getters:
    - summary: get_pm_loop
      description: Get phasemeter loop frequency and bandwidth
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
