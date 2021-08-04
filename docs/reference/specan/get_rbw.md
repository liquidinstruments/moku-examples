---
title: get_rbw | Spectrum Analyzer
additional_doc: null
description: Returns the current resolution bandwidth (Hz)
method: get
name: get_rbw
parameters: []
summary: get_rbw
---



<headers/>
<parameters/>


Examples,

<code-group>
<code-block title="Python">
```python{5}
from moku.instruments import SpectrumAnalyzer

i = SpectrumAnalyzer('192.168.###.###', force_connect=False)
i.set_rbw('Auto')  # Auto-mode
print(i.get_rbw())
```
</code-block>

<code-block title="MATLAB">
```matlab{5,6}
m = MokuSpectrumAnalyzer('192.168.###.###', false);
% Configure the spectrum analyzer to measure a span from 10Hz to 10MHz,
% auto mode, BlackmanHarris window, and no video filter
m.sa_measurement(1, 10, 10e6, 'rbw','Auto')
disp(m.get_rbw());
```
</code-block>
</code-group>