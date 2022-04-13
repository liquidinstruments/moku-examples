---
title: set_defaults | Waveform Generator
additional_doc: null
description: Set the Waveform Generator to its default state.
method: post
name: set_defaults
parameters: []
summary: set_defaults
---


<headers/>

<parameters/>

<code-group>
<code-block title="Python">
```python
from moku.instruments import WaveformGenerator

i = WaveformGenerator('192.168.###.###')
i.set_defaults()
# Variable i referring to WaveformGenerator is now in default state
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuWaveformGenerator('192.168.###.###');
m.set_defaults()
% Variable m referring to WaveformGenerator is in default state
```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{}'\
        http://<ip>/api/waveformgenerator/set_defaults
```
</code-block>

</code-group>

