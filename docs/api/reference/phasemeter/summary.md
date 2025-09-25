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

```python{5}
from moku.instruments import Phasemeter
i = Phasemeter('192.168.###.###', force_connect=True)
print(i.summary())
```

</code-block>

<code-block title="MATLAB">

```matlab{5}
m = MokuPhasemeter('192.168.###.###', force_connect=true);
disp(m.summary())
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        http://<ip>/api/phasemeter/summary
```

</code-block>

</code-group>

### Sample response

```text
Moku:Pro Phasemeter
(Input 1, on), AC coupling, 50 Ohm impedance, 4 Vpp range, set frequency 1.000 000 000 000 MHz, auto-acquire off, 1 kHz bandwidth
(Input 2, on), AC coupling, 50 Ohm impedance, 4 Vpp range, set frequency 1.000 000 000 000 MHz, auto-acquire off, 1 kHz bandwidth
(Input 3, off), AC coupling, 50 Ohm impedance, 4 Vpp range, set frequency 1.000 000 000 000 MHz, auto-acquire off, 1 kHz bandwidth
(Input 4, off), AC coupling, 50 Ohm impedance, 4 Vpp range, set frequency 1.000 000 000 000 MHz, auto-acquire off, 1 kHz bandwidth
Acquisition rate: 1.4901161194e+02 Hz
Advanced settings: phase wrapping at Off, phase auto-reset at Off
Output 1 (off) - Sine wave, 1.000 000 000 000 MHz, 500 mVpp, 0.000 000 deg, 0.000 0 V offset
Output 2 (off) - Sine wave, 1.000 000 000 000 MHz, 500 mVpp, 0.000 000 deg, 0.000 0 V offset
Output 3 (off) - Sine wave, 1.000 000 000 000 MHz, 500 mVpp, 0.000 000 deg, 0.000 0 V offset
Output 4 (off) - Sine wave, 1.000 000 000 000 MHz, 500 mVpp, 0.000 000 deg, 0.000 0 V offset
Internal 10 MHz clock
```
