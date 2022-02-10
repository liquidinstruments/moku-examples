---
title: disable_output | Spectrum Analyzer
additional_doc: null
description: Turn waveform generator output off
method: post
name: disable_output
parameters:
- default: null
  description: Target channel
  name: channel
  param_range:
   mokugo: 1, 2
   mokupro: 1, 2, 3, 4
  type: integer
  unit: null
- default: true
  description: Disable all implicit conversions and coercions.
  name: strict
  param_range: null
  type: boolean
  unit: null
summary: disable_output
---



<headers/>
<parameters/>


### Examples


<code-group>
<code-block title="Python">
```python{5,6}
from moku.instruments import SpectrumAnalyzer

i = SpectrumAnalyzer('192.168.###.###', force_connect=False)
i.set_span(frequency1=0, frequency2=30e3)
# Disable output channel 1
i.disable_output(1)
```
</code-block>

<code-block title="MATLAB">
```matlab{5,6}
m = MokuSpectrumAnalyzer('192.168.###.###', false);
% Configure the spectrum analyzer to measure a span from 10Hz to 10MHz,
% auto mode, BlackmanHarris window, and no video filter
m.sa_measurement(1, 10, 10e6, 'rbw','Auto')
% Disable output channel 1
m.disable_output(1)
```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"channel": 1}'\
        http://<ip>/api/spectrumanalyzer/disable_output
```
</code-block>


</code-group>