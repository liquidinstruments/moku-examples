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


Usage in clients, 

<code-group>
<code-block title="Python">
```python
from moku.instruments import ArbitraryWaveformGenerator
i = ArbitraryWaveformGenerator('192.168.###.###', force_connect=False)
print(i.summary())
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuArbitraryWaveformGenerator('192.168.###.###', true);
disp(m.summary());
```
</code-block>
</code-group>

Sample response,

```plaintext
Moku:Go Arbitrary Waveform Generator
Output 1: Custom waveform, 100 points, Interpolation: None
Frequency 10.000 000 kHz, amplitude 1000.0 mVpp, offset 0.000 0 V, 
phase 0.000 deg
Output 2: Custom waveform, 100 points, Interpolation: None
Frequency 10.000 000 kHz, amplitude 1000.0 mVpp, offset 0.000 0 V, 
phase 0.000 deg
```