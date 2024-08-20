---
additional_doc: null
description: Set output signal offset
method: post
name: set_output_limit
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
    - default: false
      description: Enable voltage limiter
      name: enable
      param_range: null
      type: boolean
      unit: null
    - default: null
      description: Low voltage limit
      name: low_limit
      param_range:
          mokugo: -5 to 5
          mokulab: -1 to 1
          mokupro: -1 to 1
      type: number
      unit: V
    - default: null
      description: High voltage limit
      name: high_limit
      param_range:
          mokugo: -5 to 5
          mokulab: -1 to 1
          mokupro: -1 to 1
      type: number
      unit: V
    - default: true
      description: Disable all implicit conversions and coercions.
      name: strict
      param_range: null
      type: boolean
      unit: null
summary: set_output_limit
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
# Set output limit to 1VDC
i.set_output_limit(1, low_limit=-0.5, high_limit=0.5)
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuPIDController('192.168.###.###');
% Configure the Channel 2 PID Controller using gain characteristics
%   Overall Gain = 6dB
%   I Gain       = 20dB
m.set_by_gain_and_section(2, 'overall_gain', 6.0, 'prop_gain', 20)
% Set output limit to 1VDC
m.set_output_limit(1, 'low_limit', -0.5, 'high_limit', 0.5);
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"channel": 1, "low_limit": -0.5, "high_limit": 0.5}'\
        http://<ip>/api/pidcontroller/set_output_limit
```

</code-block>

</code-group>

### Sample response

```json
{ "low_limit": -0.5, "high_limit": 0.5 }
```
