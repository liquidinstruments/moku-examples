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
from moku.instruments import LaserLockBox
i = LaserLockBox('192.168.###.###', force_connect=True)
i.set_aux_oscillator()
# Set the probes to monitor Output 1 and Output 2
i.set_monitor(1, 'Output1')
i.set_monitor(2, 'Output2')
print(i.summary())
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuLaserLockBox('192.168.###.###', force_connect=true);
m.set_aux_oscillator();
% Set the probes to monitor Output 1 and Output 2
m.set_monitor(1, 'Output1')
m.set_monitor(2, 'Output2')
disp(m.summary())
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        http://<ip>/api/laserlockbox/summary
```

</code-block>

</code-group>

### Sample response

```plaintext
Moku:Pro Laser Lock Box
Input 1, DC , 600 MHz , 1 MOhm , 0 dB , 400 mVpp, 0 dB digital gain
Input 2, DC , 600 MHz , 1 MOhm , 0 dB , 400 mVpp
Demodulation: Modulation signal, phase shift 0.000 000 deg
Lowpass filter: 4th-order Butterworth Lowpass, corner frequency 1.000 MHz, sampling rate 78.125 MHz
Setpoint: 0.000 0 V
Fast controller: PI controller: proportional gain -10.0 dB, integrator crossover 3.100 kHz, integrator saturation +40.0 dB, invert off
Slow controller: PI controller: proportional gain -10.0 dB, integrator crossover 49.00 Hz, integrator saturation +40.0 dB, invert off
Scan - Positive ramp, 10.000 000 Hz, 500 mVpp, Output 1
Modulation - 1.000 000 000 000 MHz, 500 mVpp, Output 1
Output 1 - control signal disabled, offset 0.000 0 V, limits disabled, 0 dB output gain, output disabled
Output 2 - control signal disabled, offset 0.000 0 V, limits disabled, 0 dB output gain, output disabled
External 10 MHz clock
```
