---
additional_doc: Precision mode is also known as decimation, it samples at the full rate and averages        excess data points to improve precision. Normal mode works by direct down sampling,     throwing away extra data points.
description: Changes acquisition mode between 'Normal' and 'Precision'. 
method: post
name: set_acquisition_mode
parameters:
- default: Normal
  description: Acquisition Mode
  name: mode
  param_range: Normal, Precision, PeakDetect
  type: string
  unit: null
- default: true
  description: Disable all implicit conversions and coercions.
  name: strict
  param_range: null
  type: boolean
  unit: null
summary: set_acquisition_mode
---

<headers/>
<parameters/>

### Examples

<code-group>
<code-block title="Python">
```python
from moku.instruments import Datalogger
i = Datalogger('192.168.###.###', force_connect=False)
# Generate Sine wave on Output1
i.generate_waveform(channel=1, type='Sine', amplitude=1, frequency=10e3)
i.set_acquisition_mode(mode='Precision')
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuDatalogger('192.168.###.###', true);
% Generate a sine wave on Channel 1
% 1Vpp, 10kHz, 0V offset
m.generate_waveform(1, 'Sine', 'amplitude',1, 'frequency',10e3);
% Generate a square wave on Channel 2
% 1Vpp, 10kHz, 0V offset, 50% duty cycle
m.generate_waveform(2, 'Square', 'amplitude',1, 'frequency', 1e3, 'duty', 50);
m.set_acquisition_mode('mode', 'Precision');
```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"mode": "Precision"}'\
        http://<ip>/api/datalogger/set_acquisition_mode
```
</code-block>

</code-group>