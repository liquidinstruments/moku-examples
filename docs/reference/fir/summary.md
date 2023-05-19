---
additional_doc: null
description: Returns a short summary of current instrument state
method: get
name: summary
parameters: []
summary: summary
---





<headers/>
<parameters/>

### Examples

<code-group>
<code-block title="Python">
```python
from moku.instruments import FIRFilterBox
i = FIRFilterBox('192.168.###.###', force_connect=False)
# Configure the Channel 1 PID Controller using frequency response
# characteristics
# 	P = -10dB
i.set_by_frequency(channel=1, prop_gain=-10)
# Set the probes to monitor Output 1 and Output 2
i.set_monitor(1, 'Output1')
i.set_monitor(2, 'Output2')
print(i.summary())
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuFIRFilterBox('192.168.###.###', true);
% Configure the Channel 1 FIRFilterBox using frequency response
% characteristics
% 	P = -10dB
m.set_by_frequency(1, 'prop_gain', -20);
% Set the probes to monitor Output 1 and Output 2
m.set_monitor(1, 'Output1')
m.set_monitor(2, 'Output2')
disp(m.summary())
```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        http://<ip>/api/pid/summary
```
</code-block>


</code-group>
Sample response,

```plaintext
This needs to be updated!!!!! for FIRFilterBox
Moku:Go PID Controller
Input 1 - DC coupling, 0 dB attenuation
Input 2 - DC coupling, 0 dB attenuation
Control matrix: 1-1 = 1, 1-2 = 0, 2-1 = 1, 2-2 = 0
Controller 1: PID controller: proportional gain -10.0 dB, integrator crossover 100.0 Hz, differentiator crossover 10.00 kHz, integrator saturation +10.0 dB, differentiator saturation +10.0 dB, input offset 0.000 0 V, output offset 0.000 0 V
Controller 2: PI controller: proportional gain -10.0 dB, integrator crossover 310.0 Hz, integrator saturation +40.0 dB, input offset 0.000 0 V, output offset 0.000 0 V
```