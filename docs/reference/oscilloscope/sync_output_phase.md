---
additional_doc: null
description: Synchronize the phase of both output channels.
method: post
name: sync_output_phase
parameters: []
summary: sync_output_phase
group: Embedded Waveform Generator
---


<headers/>
<parameters/>

<code-group>
<code-block title="Python">
```python{10}
from moku.instruments import Oscilloscope
# Configure the instrument
# Generate a sine wave on Channel 1
# 1Vpp, 10kHz, 0V offset
i.generate_waveform(1, 'Sine', amplitude=0.5, frequency=10e3);

# Generate a square wave on Channel 2
# 1Vpp, 10kHz, 0V offset, 50% duty cycle
i.generate_waveform(2, 'Square', amplitude=1, frequency=20e3, duty=50);
i.sync_output_phase()
```
</code-block>

<code-block title="MATLAB">
```matlab{10}
m = MokuOscilloscope('192.168.###.###', true);
%% Configure the instrument
% Generate a sine wave on Channel 1
% 1Vpp, 10kHz, 0V offset
m.generate_waveform(1, 'Sine', 'amplitude',0.5, 'frequency', 10e3);

% Generate a square wave on Channel 2
% 1Vpp, 10kHz, 0V offset, 50% duty cycle
m.generate_waveform(2, 'Square', 'amplitude',1, 'frequency',20e3, 'duty', 50);
m.sync_output_phase()
```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{}'\
        http://<ip>/api/oscilloscope/sync_output_phase
```
</code-block>

</code-group>