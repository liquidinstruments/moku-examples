---
title: summary | Waveform Generator
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
from moku.instruments import WaveformGenerator

i = WaveformGenerator('192.168.###.###')
i.generate_waveform(channel=1, type='Sine', amplitude=0.5, frequency=5e3)
i.generate_waveform(channel=2, type='Sine', amplitude=1.0, frequency=1e6)
print(i.summary())
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuWaveformGenerator('192.168.###.###');
% Generate a sine wave on Channel 1
% 1Vpp, 10kHz, 0V offset
m.generate_waveform(1, 'Sine','amplitude', 1, 'frequency',1000,'offset',0.2);
disp(m.summary());
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        http://<ip>/api/waveformgenerator/summary
```

</code-block>

</code-group>

### Sample response

```text
Moku:Go Waveform Generator
Output 1 - 5.000 000 000 kHz sine, 500.0 mVpp, offset 0.000 0 V, phase 0.000 deg
Output 2 -  sine, 1000.0 mVpp, offset 0.000 0 V, phase 0.000 deg; sweep mode, Input 1 trigger (100 mV), stop frequency 10.00 Hz, sweep time 3.000 s
```
