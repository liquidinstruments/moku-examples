---
additional_doc: If Theta is selected, the scaling is 1V per cycle.
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
    - default: null
      description: Main output DC offset
      name: main_offset
      type: number
      unit: V
    - default: null
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

:::tip Polar converter
If R or Theta is selected, then a Rectangular-to-Polar converter will be engaged. The performance of this converter can be optimised by setting the expected input range, see [set_polar_mode](./set_polar_mode)
:::

:::warning Polar/Rectangular Outputs
Only one of Polar or Rectangular outputs can be used at once. For example, it's invalid to request Main output `X` and Aux output `Theta`.
:::

<parameters/>

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import LockInAmp
i = LockInAmp('192.168.###.###')
i.set_outputs(main="X", main_offset=1)
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuLockInAmp('192.168.###.###');
m.set_outputs('main','X','main_offset',1);
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"main": "X", "main_offset": 1}'\
        http://<ip>/api/lockinamp/set_outputs
```

</code-block>

</code-group>

### Sample response

```json
{
    "aux_offset": 0.0,
    "main_offset": 1.0
}
```
