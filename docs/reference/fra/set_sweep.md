---
additional_doc: null
description: Set the output sweep parameters
method: post
name: set_sweep
parameters:
- default: 0
  description: Sweep start frequency
  name: start_frequency
  param_range: 
   mokugo: 10e-3 to 20e6
   mokupro: 10e-3 to 300e6
  type: number
  unit: Hz
- default: 0
  description: Sweep stop frequency
  name: stop_frequency
  param_range: 
   mokugo: 10e-3 to 20e6
   mokupro: 10e-3 to 300e6
  type: number
  unit: Hz
- default: 512
  description: Number of points in the sweep (rounded to nearest power of 2)
  name: num_points
  param_range: null
  type: integer
  unit: null
- default: 0
  description: Minimum averaging time per sweep point.
  name: averaging_time
  param_range: 1e-6 to 10
  type: number
  unit: Seconds
- default: 0
  description: Minimum averaging cycles per sweep point.
  name: averaging_cycles
  param_range: 1 to 1048576
  type: integer
  unit: null
- default: 0
  description: Minimum settling time per sweep point.
  name: settling_time
  param_range: 1e-6 to 10
  type: number
  unit: Seconds
- default: 0
  description: Minimum settling cycles per sweep point.
  name: settling_cycles
  param_range: 1 to 1048576
  type: integer
  unit: null
- default: false
  description: Enables linear scale. If set to false scale is set to logarithmic.
    Defaults to false
  name: linear_scale
  param_range: null
  type: boolean
  unit: null
- default: true
  description: Disable all implicit conversions and coercions.
  name: strict
  param_range: null
  type: boolean
  unit: null
summary: set_sweep
---

<headers/>
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
```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"start_frequency": 10e6, "stop_frequency": 100, "num_points": 512, "averaging_time": 10e-3, "settling_time": 10e-3, "averaging_cycles": 1, "settling_cycles": 1}'\
        http://<ip>/api/fra/set_sweep
```
</code-block>

</code-group>

### Sample response
```json
{
  "averaging_cycles":5,
  "averaging_time":0.001,
  "num_points":256,
  "settling_cycles":5,
  "settling_time":0.001,
  "start_frequency":20000000.0,
  "stop_frequency":100.0
}
```