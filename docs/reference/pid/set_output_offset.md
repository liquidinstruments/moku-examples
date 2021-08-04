---
additional_doc: null
description: Set output signal offset
method: post
name: set_output_offset
parameters:
- default: null
  description: Target channel
  name: channel
  param_range:
   mokugo: 1, 2
   mokupro: 1, 2, 3, 4
  type: integer
  unit: null
- default: null
  description: Output DC offset
  name: offset
  param_range: -5 to 5
  type: number
  unit: V
- default: true
  description: Disable all implicit conversions and coercions.
  name: strict
  param_range: null
  type: boolean
  unit: null
summary: set_output_offset
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
# Set output offset to 10Vpp
i.set_output_offset(1, offset=5)
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuPIDController('192.168.###.###', true);
% Configure the Channel 2 PID Controller using gain characteristics
%   Overall Gain = 6dB
%   I Gain       = 20dB 
m.set_by_gain_and_section(2, 'overall_gain', 6.0, 'prop_gain', 20)
% Set output offset to 10Vpp
m.set_output_offset(1, 'offset', 5);
```
</code-block>
</code-group>