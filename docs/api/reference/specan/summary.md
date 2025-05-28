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

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import SpectrumAnalyzer
i = SpectrumAnalyzer('192.168.###.###')
print(i.summary())
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuSpectrumAnalyzer('192.168.###.###');
disp(m.summary())
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        http://<ip>/api/spectrumanalyzer/summary
```

</code-block>

</code-group>

### Sample response

```plaintext
Moku:Go Spectrum Analyzer
Input 1 (on), DC coupling, 10 Vpp range
Input 2 (off), DC coupling, 10 Vpp range
Start freq 0 Hz, stop freq 10.000 000 MHz, RBW 48.88 kHz (Auto mode), Blackman-Harris window, Video filter off
Averaging: 32 spectra per frame, 1 frame averages
Output 1 (off) - 10.000 000 000 000 MHz, 1.000 Vpp
Output 2 (off) - 20.000 000 000 000 MHz, 1.000 Vpp
```
