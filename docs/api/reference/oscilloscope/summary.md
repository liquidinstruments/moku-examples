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
from moku.instruments import Oscilloscope
i = Oscilloscope('192.168.###.###')
# Configure the instrument
i.generate_waveform(1, 'Sine', amplitude=0.5, frequency=10e3)
i.generate_waveform(2, 'Square', amplitude=1, frequency=20e3, duty=50)
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

### Sample response

```text
Moku:Go Oscilloscope
Channel A (Input 1, on/off), DC coupling, 1 MOhm impedance, 1 x probe attenuation
Channel B (Input 2, on/off), DC coupling, 1 MOhm impedance, 1 x probe attenuation
Time span 10.00 us, time offset 0.000 s, Precision acquisition mode, no averaging
Edge trigger: Input 1, level 0.000 V, Auto mode, Rising edge, Auto sensitivity, noise reject off, HF reject off
Output 1 (on) - Sine, 10.000 000 000 kHz, 500.0 mVpp, offset 0.000 0 V, phase 0.000 000 deg
Output 2 (on) - Square, 20.000 000 000 kHz, 1.000 0 Vpp, offset 0.000 0 V, phase 0.000 000 deg, duty cycle 50.00 %
```
