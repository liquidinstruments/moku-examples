---
additional_doc:
    Select the main, auxiliary and PID controller sources to configure the outputs
    of the Lock-In Amplifier.
description: Configures output sources and offsets
method: post
name: set_outputs
parameters:
    - default: null
      description: Source for the Main LIA output
      name: main
      param_range: X, Y, R, Theta, Offset, None
      type: string
      unit: null
    - default: null
      description: Source for the Auxiliary LIA output
      name: aux
      param_range: Y, Theta, Demod, Aux, Offset, None
      type: string
      unit: null
    - default: undefined
      description: Main output DC offset
      name: main_offset
      type: number
      unit: V
    - default: undefined
      description: Aux output DC offset
      name: aux_offset
      type: number
      unit: V
    - default: true
      description: Disable all implicit conversions and coercions.
      name: strict
      param_range: null
      type: boolean
      unit: null
summary: set_outputs
---

<headers/>

:::warning Polar/Rectangular Outputs
Only one of Polar or Rectangular outputs can be used at once. For example, it's invalid
to request the `main` output as `X` and `aux` output as `Theta`.
:::

:::tip Polar converter
If `R` or `Theta` is selected, then a Rectangular-to-Polar converter will be engaged. If
Theta is selected, the scaling range is 1 V / cycle. The performance of this converter
can be optimised by setting the expected input range, see [set_polar_mode](./set_polar_mode.md)
:::

The Lock-in Amplifier can output a `main` and `aux` (auxiliary) output, of which many output
sources are available. The output sources selected in `set_outputs` are available
depending on the Lock-in Amplifier configuration, consisting of the demodulation
source (set with [set_demodulation](./set_demodulation.md)), auxiliary output (set
with [set_aux_output](./set_aux_output.md)) and the PID controller (set with
[use_pid](./use_pid.md) and [set_by_frequency](./set_by_frequency.md)). Set the
Lock-in Amplifier configuration, then connect the sources to an output with set_outputs.

When using `Internal` and `External PLL` [set_demodulation](./set_demodulation.md) modes,
the PID controller [use_pid](./use_pid.md) can always be attached to the `main`
output. However, the PID controller can be attached to either the `main` or `aux`
sources when set_outputs is not outputting a demodulation source (`Demod`) or an
auxiliary oscillator (`Aux`).

When using `External` and `None` [set_demodulation](./set_demodulation.md) modes, the PID
controller can only be attached to the `main` output using [use_pid](./use_pid.md). Further,
when using these demodulation modes, the `aux` output can only be set to `Aux` oscillator.

<parameters/>

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import LockInAmp
i = LockInAmp('192.168.###.###')
i.set_outputs(main="X", main_offset=1, aux="None")
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuLockInAmp('192.168.###.###');
m.set_outputs('main','X','main_offset',1, 'aux', 'None');
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"main": "X", "main_offset": 1, "aux": "None"}'\
        http://<ip>/api/lockinamp/set_outputs
```

</code-block>

</code-group>

### Sample response

```json
{
    "Auxiliary output": 2,
    "Enabled": False,
    "Polar conversion": False,
    "Signal enabled": True,
    "Use quadrature component": False,
    "aux_offset": 0.0,
    "main_offset": 1.0
}
```
