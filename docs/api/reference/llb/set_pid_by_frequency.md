---
additional_doc: null
description: Configure the selected PID controller using crossover frequencies. Channel 1 configures the fast controller, channel 2 configures the slow controller.
method: post
name: set_pid_by_frequency
parameters:
    - default: null
      description: Target channel
      name: channel
      param_range: 1, 2
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
i = LaserLockBox('192.168.###.###', force_connect=True)
# Configure the fast PID controller with -10 dB proportional gain,
# a 3.1 kHz integrator crossover, and +40 dB saturation
i.set_pid_by_frequency(channel=1, prop_gain=-10, int_crossover=3_100,
                       int_saturation=40)
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuLaserLockBox('192.168.###.###', force_connect=true);
% Configure the fast PID controller with -10 dB proportional gain,
% a 3.1 kHz integrator crossover, and +40 dB saturation
m.set_pid_by_frequency(1, 'prop_gain', -10, 'int_crossover', 3.1e3, ...
                       'int_saturation', 40);
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"channel": 1, "prop_gain": -10, "int_crossover": 3100, 
                 "int_saturation": 40}'\
        http://<ip>/api/laserlockbox/set_pid_by_frequency
```

</code-block>

</code-group>

### Sample response

```json
{
    "int_crossover": 3100.0,
    "int_saturation": 40.0,
    "invert": false,
    "prop_gain": -10.0
}
```
