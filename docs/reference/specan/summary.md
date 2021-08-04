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
from moku.instruments import SpectrumAnalyzer
i = SpectrumAnalyzer('192.168.###.###', force_connect=False)
print(i.summary())
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuSpectrumAnalyzer('192.168.###.###', true);
disp(m.summary());
```
</code-block>
</code-group>

Sample response,

```plaintext
Moku:Go Spectrum Analyzer
Input 1 - DC coupling, 10 Vpp range
Input 2 - DC coupling, 10 Vpp range
Start freq 0 Hz, stop freq 10.000 00 MHz, RBW 48.88 kHz (Auto mode), Blackman-Harris window, 
Video filter off, no averaging
```