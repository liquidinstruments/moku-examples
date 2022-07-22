---
additional_doc: null
description: Start sweeping
method: post
name: start_sweep
parameters:
- default: false
  description: If true, enables single sweep mode.
  name: single
  param_range: null
  type: boolean
  unit: null
  warning: To avoid timeouts while reading the data, it is recommended to set the timeout parameter of `get_data` function to a value greater than the `estimated_sweep_time` which is included in the response of this function.
- default: true
  description: Disable all implicit conversions and coercions.
  name: strict
  param_range: null
  type: boolean
  unit: null
summary: start_sweep
---

<headers/>
Start sweeping
<parameters/>

<code-group>
<code-block title="Python">
```python
from moku.instruments import FrequencyResponseAnalyzer

i = FrequencyResponseAnalyzer('192.168.###.###')
# Measure input signal on channel 1
i.fra_measurement(1, input_only=True, start_frequency=100,
                  stop_frequency=20e6, averaging_cycles=1)
# Set output sweep configuration
# 10MHz - 100Hz, 512 sweep points
# Logarithmic sweep ON
# 1msec averaging time, 1msec settling time
# 1 averaging cycle, 1 settling cycle
i.set_sweep(start_frequency=10e6, stop_frequency=100,
      num_points=512, averaging_time=10e-3,
      settling_time=10e-3, averaging_cycles=1,
      settling_cycles=1)
i.start_sweep()      
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuFrequencyResponseAnalyzer('192.168.###.###');
% Set output sweep configuration
% 10MHz - 100Hz, 512 sweep points
% Logarithmic sweep ON
% 1msec averaging time, 1msec settling time
% 1 averaging cycle, 1 settling cycle
m.set_sweep('start_frequency', 10e6, 'stop_frequency', 100, 'num_points', 512, ...
    'averaging_time', 10e-3, 'averaging_cycles', 1,...
    'settling_time', 10e-3, 'settling_cycles', 1);
m.start_sweep()
```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{}'\
        http://<ip>/api/fra/start_sweep
```
</code-block>

</code-group>

### Sample response
```json
{
 "estimated_sweep_time": 10.492993950848678
}
```