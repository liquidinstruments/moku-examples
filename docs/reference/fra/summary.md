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
i = FrequencyResponseAnalyzer('192.168.###.###', force_connect=False)
print(i.summary())
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuFrequencyResponseAnalyzer('192.168.###.###', true);
disp(m.summary());
```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{}'\
        http://<ip>/api/fra/summary
```
</code-block>

</code-group>

Sample response,

```plaintext
Moku:Go Frequency Response Analyzer
Channel A - DC coupling, 10 Vpp range, amplitude 100 mVpp, offset 0.000 0 V, phase 0.000 deg
Channel B - DC coupling, 10 Vpp range, amplitude 100 mVpp, offset 0.000 0 V, phase 0.000 deg
Logarithmic sweep, measuring fundamental, calibration off
Averaging time 1.000 ms, 5 cycles; Settling time 1.000 ms, 5 cycles
```