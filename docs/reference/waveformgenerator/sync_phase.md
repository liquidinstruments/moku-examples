---
title: sync_phase | Waveform Generator
additional_doc: The phase of all channels is reset to their respective phase offset values
description: Synchronize the phase of all output channels. 
method: get
name: sync_phase
parameters: []
summary: sync_phase
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
# Phase sync between the two channels
i.sync_phase()
```
</code-block>

<code-block title="MATLAB">
```matlab

m = MokuWaveformGenerator('192.168.###.###');
% Generate a sine wave on Channel 1
% 1Vpp, 10kHz, 0V offset
m.generate_waveform(1, 'Sine','amplitude', 1, 'frequency',1000,'offset',0.2);
% Generate a sine wave on Channel 2
% 1Vpp, 10kHz, 0V offset, 50% duty cycle
m.generate_waveform(2, 'Sine', 'amplitude',1,'frequency', 10e3);
% Phase sync between the two channels
m.sync_phase();
```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        http://<ip>/api/waveformgenerator/sync_phase
```
</code-block>

</code-group>
