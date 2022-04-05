---
additional_doc: null
description: Sets up commonly used configurations in the Frequency Response Analyzer,
  including the frequency range, averaging, and swept sine amplitude.
method: post
name: fra_measurement
parameters:
- default: null
  description: Target channel
  name: channel
  param_range:
   mokugo: 1, 2
   mokupro: 1, 2, 3, 4
  type: integer
  unit: null
- default: false
  description: If enabled, measures input signal alone. Defaults to false, which is
    In/Out mode
  name: input_only
  param_range: null
  type: boolean
  unit: null
- default: 0
  description: Sweep start frequency
  name: start_frequency
  param_range: 
   mokugo: 10e-3 to 20e6
   mokupro: 10e-3 to 300e6
  type: number
  unit: Hz
- default: 0
  description: Sweep end frequency
  name: stop_frequency
  param_range: 
   mokugo: 10e-3 to 20e6
   mokupro: 10e-3 to 300e6
  type: number
  unit: Hz
- default: 0
  description: Minimum averaging time per sweep point.
  name: averaging_duration
  param_range: 1e-6 to 10
  type: number
  unit: Seconds
- default: 0
  description: Minimum averaging cycles per sweep point.
  name: averaging_cycles
  param_range: 1 to 1,048,576
  type: integer
  unit: null
- default: 0
  description: Output amplitude (For Moku:Pro, the output voltage is limited to between - 1V and 1 V above 1 MHz)
  name: output_amplitude 
  param_range: 
   mokugo: 2e-3 to 10
   mokupro: 1e-3 to 10
  type: number
  unit: Vpp
- default: true
  description: Disable all implicit conversions and coercions.
  name: strict
  param_range: null
  type: boolean
  unit: null
summary: fra_measurement
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
                  stop_frequency=20e6, averaging_duration=1, 
                  averaging_cycles=1, output_amplitude=0.001)
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuFrequencyResponseAnalyzer('192.168.###.###');
% Measure input signal on channel 1
m.fra_measurement(1, 'input_only', true, 'start_frequency', 100,
                  'stop_frequency', 20e6, 'averaging_duration', 1,
                  'averaging_cycles', 1, 'output_amplitude', 0.001)
```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"channel": 1, "input_only": true, "start_frequency": 100, "stop_frequency": 20e6,"averaging_duration": 1,"averaging_cycles": 1, "output_amplitude":0.001}'\
        http://<ip>/api/fra/fra_measurement
```
</code-block>

</code-group>

### Sample response
```json
{
  "averaging_cycles":5,
  "averaging_duration":0.001,
  "output_amplitude":1.0,
  "start_frequency":20000000.0,
  "stop_frequency":100.0}
}
```
