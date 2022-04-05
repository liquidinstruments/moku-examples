---
additional_doc: null
description: Configures the specified monitor channel to view the desired PID instrument
  signal.
method: post
name: set_monitor
parameters:
- default: true
  description: Disable all implicit conversions and coercions.
  name: strict
  param_range: null
  type: boolean
  unit: null
- default: null
  description: Monitor channel
  name: monitor_channel
  param_range: null
  type: integer
  unit: null
- default: null
  description: Monitor channel source. 
  name: source
  param_range: None, Input1, Control1, Output1, Input2, Control2, Output2
  type: string
  unit: null
summary: set_monitor
group: Monitors
---

<headers/>

There are two monitoring channels available, each of these can be assigned
to source signals from any of the internal PID instrument monitoring points. Signals
larger than 12-bits must be either truncated or clipped to the allowed size.

Source signal can be one of,
 - Input1 : Channel 1 ADC input
 - Control1 : PID Channel 1 input (after mixing, offset and scaling)
 - Output1 : PID Channel 1 output
 - Input2 : Channel 2 ADC Input
 - Control2 : PID Channel 2 input (after mixing, offset and scaling)
 - Output2 : PID Channel 2 output

<parameters/>

### Examples

<code-group>
<code-block title="Python">
```python
from moku.instruments import PIDController
i = PIDController('192.168.###.###')
# Configure the Channel 1 PID Controller using frequency response
# characteristics
# 	P = -10dB
i.set_by_frequency(channel=1, prop_gain=-10)
# Set the probes to monitor Output 1 and Output 2
i.set_monitor(1, 'Output1')
i.set_monitor(2, 'Output2')

```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuPIDController('192.168.###.###');
% Configure the Channel 1 PID Controller using frequency response
% characteristics
% 	P = -10dB
m.set_by_frequency(1, 'prop_gain', -20);
% Set the probes to monitor Output 1 and Output 2
m.set_monitor(1, 'Output1')
m.set_monitor(2, 'Output2')
```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"channel": 1, "source": "Output1"}'\
        http://<ip>/api/pidcontroller/set_monitor
```
</code-block>

</code-group>

### Sample response,

```json
{
  "source": "Output1"
}
```