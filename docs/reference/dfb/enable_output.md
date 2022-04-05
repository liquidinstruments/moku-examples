---
additional_doc: The Digital Filter Box instrument has an output stage between the filter itself and the
    hardware outputs. This output stage includes the output offset and may contain a hardware gain stage
    depending on the Moku hardware platform (e.g. Moku:Pro has a selectable +14dB hardware driver required
    to obtain maximum output range).

    To have a filter output present on the device, the signal must be connected to the output stage (signal=true)
    and the output stage must also be enabled (output=true).
description: Enable or disable the Digital Filter output(s)
method: post
name: enable_output
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
  description: Connect/Disconnect the filter signal to/from the output stage
  name: signal
  param_range: null
  type: boolean
  unit: null
- default: null
  description: Enable/Disable output driver
  name: output
  param_range: null
  type: boolean
  unit: null
- default: "0dB"
  description: If applicable, enable the hardware's output gain stage (high output range)
  name: gain_range
  param_range: 0dB, 14dB
  type: string
  unit: dB
- default: true
  description: Disable all implicit conversions and coercions.
  name: strict
  param_range: null
  type: boolean
  unit: null
summary: enable_output
---

<headers/>
<parameters/>


### Examples

<code-group>
<code-block title="Python">
```python
from moku.instruments import DigitalFilterBox
i = DigitalFilterBox('192.168.###.###')
# Configure the Channel 2 PID Controller using gain characteristics
#   Overall Gain = 6dB
#   I Gain       = 20dB 
i.set_by_gain(channel=2, overall_gain=6.0, int_gain=20)
# Enable the output channels of the PID controller
i.enable_output(1, True, True)
i.enable_output(2, True, True)
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuDigitalFilterBox('192.168.###.###');
% Configure the Channel 2 PID Controller using gain characteristics
%   Overall Gain = 6dB
%   I Gain       = 20dB 
m.set_by_gain_and_section(2, 'overall_gain', 6.0, 'int_gain', 20)
% Enable the output channels of the PID controller
m.enable_output(1, 'signal', True, 'output', True);
m.enable_output(2, 'signal', True, 'output', True);
```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"channel": 2, "signal": true, "output": true}'\
        http://<ip>/api/digitalfilterbox/enable_output
```
</code-block>

</code-group>

### Sample response
```json
{
  "gain_range": "0dB",
  "output": true,
  "signal": true
}
```