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

<code-group>
<code-block title="Python">
```python
from moku.instruments import Oscilloscope
# Configure the instrument
i.generate_waveform(1, 'Sine', amplitude=0.5, frequency=10e3);
i.generate_waveform(2, 'Square', amplitude=1, frequency=20e3, duty=50);
print(i.summary())
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuOscilloscope('192.168.###.###', true);
%% Configure the instrument
m.generate_waveform(1, 'Sine', 'amplitude',0.5, 'frequency', 10e3);
m.generate_waveform(2, 'Square', 'amplitude',1, 'frequency',20e3, 'duty', 50);
disp(m.summary())
```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        http://<ip>/api/oscilloscope/summary
```
</code-block>

</code-group>

Sample response,

```text
Moku:Go Oscilloscope
Channel A (Input 1) - DC coupling, 1 x probe attenuation
Channel B (Input 2) - DC coupling, 1 x probe attenuation
Time span 10.00 us, time offset 0.000 s, Precision acquisition mode, no averaging
Edge trigger: Input 1, level 0.000 V, Auto mode, Rising edge, Auto sensitivity, noise reject off, HF reject off
```