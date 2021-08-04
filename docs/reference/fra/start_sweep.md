---
additional_doc: null
description: Start sweeping
method: post
name: start_sweep
parameters: []
summary: start_sweep
---

<headers/>
Start sweeping
<parameters/>

<code-group>
<code-block title="Python">
```python
from moku.instruments import FrequencyResponseAnalyzer

i = FrequencyResponseAnalyzer('192.168.###.###', force_connect=False)
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
m = MokuFrequencyResponseAnalyzer('192.168.###.###', true);
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
</code-group>