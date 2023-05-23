---
additional_doc: The PID controller block in the Moku contains two cascaded sections. This function allows full
    configuration of both sections for maximum flexibility. For a simpler interface, see `set_by_gain` and
    `set_by_frequency`.
description: Configure the selected PID controller and its sections using gain coefficients.
method: post
name: set_by_gain_and_section
parameters:
- default: null
  description: Target channel
  name: channel
  param_range:
   mokugo: 1, 2
   mokulab: 1, 2
   mokupro: 1, 2, 3, 4
  type: integer
  unit: null
- default: null
  description: Section to configure
  name: section
  param_range: null
  type: integer
  unit: null
- default: undefined
  description: Overall Gain
  name: overall_gain
  param_range: null
  type: number
  unit: dB
- default: undefined
  description: Proportional gain factor
  name: prop_gain
  param_range: null
  type: number
  unit: dB
- default: undefined
  description: Integrator gain factor
  name: int_gain
  param_range: null
  type: number
  unit: dB
- default: undefined
  description: Differentiator gain factor
  name: diff_gain
  param_range: null
  type: number
  unit: dB
- default: undefined
  description: Integrator gain saturation corner
  name: int_corner
  param_range: null
  type: number
  unit: Hz
- default: undefined
  description: Differentiator gain saturation corner
  name: diff_corner
  param_range: null
  type: number
  unit: Hz
- default: true
  description: Disable all implicit conversions and coercions.
  name: strict
  param_range: null
  type: boolean
  unit: null
summary: set_by_gain_and_section
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
i.set_by_gain_and_section(channel=2,section=1, overall_gain=6.0, prop_gain=20)
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuPIDController('192.168.###.###');
% Configure the Channel 2 PID Controller using gain characteristics
%   Overall Gain = 6dB
%   I Gain       = 20dB 
m.set_by_gain_and_section(2, 1, 'overall_gain', 6.0, 'prop_gain', 20)
```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key:<key>'\
        -H 'Content-Type: application/json'\
        --data '{"channel": 2, "section": 1, "prop_gain": 20}'\
        http://<ip>/api/pidcontroller/set_by_gain_and_section
```
</code-block>

</code-group>

### Sample response
```json
{
  "diff_corner": 100.0,
  "diff_gain": 0.0,
  "int_corner": 5.0,
  "int_gain": 40.0,
  "overall_gain": 6.0,
  "prop_gain": 20.0
}
```