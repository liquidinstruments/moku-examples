---
additional_doc: null
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
   mokupro: 1, 2, 3, 4
  type: integer
  unit: null
- default: 20
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
  description: Integrator gain corner
  name: int_corner
  param_range: null
  type: number
  unit: Hz
- default: null
  description: Differentiator gain corner
  name: diff_corner
  param_range: null
  type: number
  unit: Hz
summary: set_by_gain
---


<headers/>
<parameters/>

Usage in clients, 

<code-group>
<code-block title="Python">
```python
from moku.instruments import PIDController
i = PIDController('192.168.###.###', force_connect=False)
# Configure the Channel 2 PID Controller using gain characteristics
#   Overall Gain = 6dB
#   I Gain       = 20dB 
i.set_by_gain(channel=2, overall_gain=6.0, prop_gain=20)
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuPIDController('192.168.###.###', true);
% Configure the Channel 2 PID Controller using gain characteristics
%   Overall Gain = 6dB
%   I Gain       = 20dB 
m.set_by_gain_and_section(2, 'overall_gain', 6.0, 'prop_gain', 20)
```
</code-block>
</code-group>