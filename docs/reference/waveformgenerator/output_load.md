---
title: output_load | Waveform Generator
additional_doc: null
description: Configure load on an output channel.
method: post
name: output_load
parameters:

- default: null
  description: Target channel
  name: channel
  param_range: 1, 2, 3, 4
  type: integer
  unit: null
- default: 1MOhm
  description: Output load
  name: load
  param_range: 50Ohm, 1MOhm
  type: string
  unit: null
summary: output_load
available_on: "mokupro"
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
# Configure load on channel 1. 
i.output_load(1, "1MOhm")
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
% Configure load on channel 1. 
m.output_load(1, '1MOhm')

```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"channel": 1, "load": "1MOhm"}'\
        http://<ip>/api/waveformgenerator/output_load
```
</code-block>

</code-group>

### Sample response 
```json
{
  "load": "1MOhm"
}
```