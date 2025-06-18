---
additional_doc: null
description: Configure the selected PID controller using crossover frequencies.
method: post
name: set_pid_by_frequency
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
    - default: undefined
      description: Proportional gain factor
      name: prop_gain
      param_range: -60 to 60
      type: number
      unit: dB
    - default: undefined
      description: Integrator crossover frequency
      name: int_crossover
      param_range: 31.25e-3 to 312.5e3
      type: number
      unit: Hz
    - default: undefined
      description: Differentiator crossover frequency
      name: diff_crossover
      param_range: 312.5e-3 to 3.125e6
      type: number
      unit: Hz
    - default: undefined
      description: Second integrator crossover frequency
      name: double_int_crossover
      param_range: 31.25e-3 to 312.5e3
      type: number
      unit: Hz
    - default: undefined
      description: Integrator gain saturation
      name: int_saturation
      param_range: -60 to 60
      type: number
      unit: dB
    - default: undefined
      description: Differentiator gain saturation
      name: diff_saturation
      param_range: -60 to 60
      type: number
      unit: dB
    - default: false
      description: Invert PID
      name: invert
      param_range: null
      type: number
      unit: null
    - default: true
      description: Disable all implicit conversions and coercions.
      name: strict
      param_range: null
      type: boolean
      unit: null
summary: set_pid_by_frequency
---

<headers/>
<parameters/>

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import LaserLockBox
i = LaserLockBox('192.168.###.###', force_connect=False)
# Configure the Fast PID Controller using frequency response
# characteristics
# 	P = -10dB
i.set_pid_by_frequency(channel=1, prop_gain=-10)
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuLaserLockBox('192.168.###.###', true);
% Configure the Fast PID Controller using frequency response
% characteristics
% 	P = -10dB
m.set_pid_by_frequency(1, 'prop_gain', -10);
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"channel": 1, "prop_gain": -10}'\
        http://<ip>/api/laserlockbox/set_pid_by_frequency
```

</code-block>

</code-group>

### Sample response

```json
{
    "diff_crossover": 63000.0,
    "diff_saturation": 15.0,
    "double_int_crossover": 130.0,
    "int_crossover": 1300.0,
    "int_saturation": 40.0,
    "prop_gain": -10.0
}
```
