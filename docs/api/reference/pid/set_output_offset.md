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
          mokulab: 1, 2
          mokupro: 1, 2, 3, 4
          mokudelta: 1, 2, 3, 4
      type: integer
      unit: null
    - default: null
      description: Output DC offset
      name: offset
      param_range:
          mokugo: -2.5 to 2.5
          mokulab: -1 to 1
          mokupro: -1 to 1
          mokudelta: -500e-3 to 500e-3
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
# Set output offset to 1VDC
i.set_output_offset(1, offset=1)
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuPIDController('192.168.###.###');
% Configure the Channel 2 PID Controller using gain characteristics
%   Overall Gain = 6dB
%   I Gain       = 20dB
m.set_by_gain_and_section(2, 'overall_gain', 6.0, 'prop_gain', 20)
% Set output offset to 1VDC
m.set_output_offset(1, 'offset', 1);
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"channel": 1, "offset": 1}'\
        http://<ip>/api/pidcontroller/set_output_offset
```

</code-block>

</code-group>

### Sample response

```json
{ "offset": 1.0 }
```
