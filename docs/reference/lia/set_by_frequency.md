---
additional_doc: The PID must also be enabled using `use_pid`
description: Configure the embedded PID controller using crossover frequencies.
method: post
name: set_by_frequency
parameters:
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
  description: Invert PID output
  name: invert
  param_range: null
  type: boolean
  unit: null
- default: true
  description: Disable all implicit conversions and coercions.
  name: strict
  param_range: null
  type: boolean
  unit: null
summary: set_by_frequency
group: PID Controller
---

<headers/>
<parameters/>


### Examples

<code-group>
<code-block title="Python">
```python
from moku.instruments import LockInAmp
i = LockInAmp('192.168.###.###')
i.set_by_frequency(prop_gain=-10)
i.use_pid(True)
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuLockInAmp('192.168.###.###');
m.set_by_frequency('prop_gain', -10);
m.use_pid(true);
```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"prop_gain": -10}'\
        http://<ip>/api/lockinamp/set_by_frequency
```
</code-block>

</code-group>

### Sample response
```json
{
  "diff_crossover": 16000.0,
  "diff_saturation": 15.0,
  "int_crossover": 310.0,
  "int_saturation": 40.0,
  "prop_gain": -10.0
}
```