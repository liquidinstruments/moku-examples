---
additional_doc: null
description: Resets the phase accumulator of both output waveforms
method: get
name: sync_phase
parameters: []
summary: sync_phase
---





<headers/>
<parameters/>

Usage in clients, 

<code-group>
<code-block title="Python">
```python
from moku.instruments import ArbitraryWaveformGenerator
i = ArbitraryWaveformGenerator('192.168.###.###', force_connect=False)
# Configure the output waveform in each channel
i.sync_phase()
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuArbitraryWaveformGenerator('192.168.###.###', true);
% Configure the output waveform in each channel
m.sync_phase();
```
</code-block>
</code-group>

