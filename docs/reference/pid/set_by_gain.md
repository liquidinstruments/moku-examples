---
additional_doc: The PID instrument contains two cascaded PID sections. Using this function sets the overall gains of the
    two sections in an optimal way. If you wish to set them individually, including configuring double-integrators, see
    `set_by_gain_and_section`.
description: Configure the selected PID controller using gain coefficients.
method: post
name: set_by_gain
parameters:
- default: true
  description: Disable all implicit conversions and coercions.
  name: strict
  param_range: null
  type: boolean
  unit: null
- default: null
  description: Target channel
  name: channel
  param_range:
   mokugo: 1, 2
   mokulab: 1, 2
   mokupro: 1, 2, 3, 4
  type: integer
  unit: null
- default: undefined
  description: Overall Gain
  name: overall_gain
  param_range: null
  type: number
  unit: dB
- default: null
  description: Proportional gain factor
  name: prop_gain
  param_range: null
  type: number
  unit: dB
- default: null
  description: Integrator gain factor
  name: int_gain
  param_range: null
  type: number
  unit: dB
- default: null
  description: Differentiator gain factor
  name: diff_gain
  param_range: null
  type: number
  unit: dB
- default: null
  description: Integrator gain saturation corner
  name: int_corner
  param_range: null
  type: number
  unit: Hz
- default: null
  description: Differentiator gain saturation corner
  name: diff_corner
  param_range: null
  type: number
  unit: Hz
summary: set_by_gain
---


<headers/>
<parameters/>

### Examples

<code-group>
<code-block title="Python">
```python
from moku.instruments import PIDController
i = PIDController('192.168.###.###')
# Configure the Channel 2 PID Controller using gain characteristics
#   Overall Gain = 6dB
#   I Gain       = 20dB 
i.set_by_gain(channel=2, overall_gain=6.0, prop_gain=20)
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuPIDController('192.168.###.###');
% Configure the Channel 2 PID Controller using gain characteristics
%   Overall Gain = 6dB
%   I Gain       = 20dB 
m.set_by_gain_and_section(2, 'overall_gain', 6.0, 'prop_gain', 20)
```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"channel": 2, "prop_gain": 20}'\
        http://<ip>/api/pidcontroller/set_by_gain
```
</code-block>

</code-group>

### Sample response
```json
[
  {
    "diff_corner": 100.0,
    "diff_gain": 0.0,
    "int_corner": 5.0,
    "int_gain": 40.0,
    "overall_gain": 6.0,
    "prop_gain": 20.0
  },
  {
    "diff_corner": 1000000.0,
    "diff_gain": -60.0,
    "int_corner": 10000.0,
    "int_gain": 60.0,
    "overall_gain": 6.0,
    "prop_gain": 20.0
  }
]
```