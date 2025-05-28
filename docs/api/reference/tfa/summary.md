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
from moku.instruments import TimeFrequencyAnalyzer
i = TimeFrequencyAnalyzer('192.168.###.###')
print(i.summary())
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuTimeFrequencyAnalyzer('192.168.###.###');
disp(m.summary())
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        http://<ip>/api/tfa/summary
```

</code-block>

</code-group>

### Sample response

```plaintext
Moku:Go Time & Frequency Analyzer
Windowed acquisition, window length 100 ms
Linear interpolation
Event A (logging on) - Input 1, 0.000 V, Rising edge, 0.000 s holdoff
Event B (logging on) - Input 2, 0.000 V, Rising edge, 0.000 s holdoff
Interval A (on) - Start: Event A, Stop: Event A
Interval B (on) - Start: Event B, Stop: Event B
Histograms - Start time 0.000 000 00 s, stop time 100.000 000 us
Multiple start events: Use first
Output 1 - Interval A, Zero point: 0.000 s, 1.000 0 kV/s, Invert off, 10 Vpp
Output 2 - Interval B, Zero point: 0.000 s, 1.000 0 kV/s, Invert off, 10 Vpp
```
