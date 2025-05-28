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
from moku.instruments import FIRFilterBox
i = FIRFilterBox('192.168.###.###', force_connect=True)
print(i.summary())
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuFIRFilterBox('192.168.###.###', true);
disp(m.summary())
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        http://<ip>/api/fir/summary
```

</code-block>

</code-group>

### Sample response

```plaintext
Moku:Go FIR Filter Builder
Input 1, DC coupling, 0 dB attenuation
Input 2, DC coupling, 0 dB attenuation
Control matrix: 1-1 = 1, 1-2 = 0, 2-1 = 0, 2-2 = 1
Filter 1 - 390.6 kHz Lowpass, Fs = 3.906 MHz, 201 taps, Window: Blackman, input offset 0.000 V, input gain +0.0 dB
Filter 2 - Impulse response: Gaussian, Fs = 488.3 kHz, 1,000 taps, Window: Hann, input offset 0.000 V, input gain +0.0 dB
Output 1 - gain +0.0 dB, offset 0.000 0 V, output disabled
Output 2 - gain +0.0 dB, offset 0.000 0 V, output disabled
```
