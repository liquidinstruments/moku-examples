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
from moku.instruments import FrequencyResponseAnalyzer
i = FrequencyResponseAnalyzer('192.168.###.###')
print(i.summary())
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuFrequencyResponseAnalyzer('192.168.###.###');
disp(m.summary())
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        http://<ip>/api/fra/summary
```

</code-block>

</code-group>

### Sample response

```plaintext
Moku:Pro Frequency Response Analyzer
Measurement mode: In / Out (dB)
Channel 1 (on), DC , 600 MHz , 1 MOhm , 4 Vpp, amplitude 100 mVpp (on), offset 0.000 0 V (off), output termination Hi-Z, phase 0.000 deg
Channel 2 (on), DC , 600 MHz , 1 MOhm , 4 Vpp, amplitude 100 mVpp (on), offset 0.000 0 V (off), output termination Hi-Z, phase 0.000 deg
Channel 3 (on), DC , 600 MHz , 1 MOhm , 4 Vpp, amplitude 100 mVpp (on), offset 0.000 0 V (off), output termination Hi-Z, phase 0.000 deg
Channel 4 (on), DC , 600 MHz , 1 MOhm , 4 Vpp, amplitude 100 mVpp (on), offset 0.000 0 V (off), output termination Hi-Z, phase 0.000 deg
Logarithmic sweep from 100.0000 kHz to 100.0000 Hz with 512 pts, dynamic amplitude mode off, measuring fundamental, delay compensation 250.0 ns (auto), normalization off
Averaging time 2.00 ms, 1 cycles; Settling time 100 us, 1 cycles
External 10 MHz clock
```
