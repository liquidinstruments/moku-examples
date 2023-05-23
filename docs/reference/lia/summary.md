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
from moku.instruments import LockInAmp
i = LockInAmp('192.168.###.###')
print(i.summary())
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuLockInAmp('192.168.###.###');
disp(m.summary());
```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        http://<ip>/api/lockinamp/summary
```
</code-block>

</code-group>

### Sample response

```json
Moku:Pro Lock-In Amplifier
Input 1, AC coupling, 1 MOhm impedance, -20 dB attenuation
Input 2, DC coupling, 50 Ohm impedance, 0 dB attenuation
Internal local oscillator: 1.000 000 000 000 MHz, 0.000 deg
Lowpass filter corner frequency 100 Hz (time constant 1.59 ms), slope 6 dB \/ octave
Main output - R signal, +10.0 dB gain, offset 1.000 0 V, 
invert off
Auxiliary output - Theta signal, +20.0 dB gain, offset 0.000 0 V, 
invert off
Internal 10 MHz clock
```