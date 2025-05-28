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
from moku.instruments import ArbitraryWaveformGenerator
i = ArbitraryWaveformGenerator('192.168.###.###')
print(i.summary())
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuArbitraryWaveformGenerator('192.168.###.###');
disp(m.summary())
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        http://<ip>/api/awg/summary
```

</code-block>

</code-group>

### Sample response

```plaintext
Moku:Go Arbitrary Waveform Generator
Output 1 (on) - Custom waveform, 100 points, Interpolation: None
Frequency 10.000 000 kHz, amplitude 1.000 0 Vpp, offset 0.000 0 V, phase 0.000 000 deg
Output 2 (on) - Custom waveform, 100 points, Interpolation: None
Frequency 10.000 000 kHz, amplitude 1.000 0 Vpp, offset 0.000 0 V, phase 0.000 000 deg
```
