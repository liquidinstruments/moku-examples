---
title: Lock-in Amplifier
prev: false
---

# Lock-in Amplifier

This instrument features a dual-phase demodulator with cascaded single pole low-pass
filters to attenuate the second harmonic and suppress noise in each quadrature

If you are directly using the RESTful API (e.g. using cURL), the instrument name as
used in the URL is `lockinamp`.

## Configuration guide

The Lock-in Amplifier configuration is made up of the:

-   Demodulation source (set with [set_demodulation](./set_demodulation.md)),
-   Auxiliary output (set with [set_aux_output](./set_aux_output.md) and [set_outputs](./set_outputs.md)), and
-   PID controller (set with [use_pid](./use_pid.md) and [set_by_frequency](./set_by_frequency.md)).

A recommended workflow is to set the main, auxiliary, and PID controller settings, then connect
the sources to an output with [set_outputs](./set_outputs.md) last, as each selection can change the
available options. Read more about the possible options in the [set_outputs](./set_outputs.md).

Set `strict=False` to move between incompatible configurations, this will coerce the instrument into
the unsetting other values as required, read more about [strict coercions](../README.md#strict-mode).
For example, when moving from using the PID controller in filtered signal mode (outputting both `R` and `Theta`
or `X` and `Y` signals) to another configuration (setting Main output as `R` and Aux output as `Aux`).

```json
Lock-in Amplifier configuration
├─── demodulation source  # Select the main demodulation source
|   ├── Internal              i.set_demodulation(mode="Internal")
|   ├── External              i.set_demodulation(mode="External")
|   ├── External PLL          i.set_demodulation(mode="External PLL")
|   └── None                  i.set_demodulation(mode="None")
|
├── auxiliary output     # Select what the signal connected to the auxiliary output
|   ├── Auxiliary oscillator      i.set_aux_output(frequency=1000, amplitude=1)
|   |                             i.set_outputs(main="X", aux="Aux")
|   |
|   ├── Demodulation              i.set_demodulation(mode="Internal", frequency=1000)
|   |                             i.set_outputs(main="X", aux="Demod")
|   |
|   └── Filtered signal           i.set_outputs(main="X", aux="Y")
|
└── PID controller       # Attach a PID controller to the main or auxiliary outputs
    ├── Off                       i.use_pid("Off")
    |
    ├── Main                      i.use_pid("Main")
    |                             i.set_by_frequency(prop_gain=-10)
    |
    └── Aux                       i.set_demodulation(mode="Internal")
                                  i.set_outputs(main="X", aux="Y")
                                  i.use_pid("Aux")
                                  i.set_by_frequency(prop_gain=-10)
```

<function-index/>
