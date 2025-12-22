---
additional_doc: Setting the bandwidth is only available on Moku:Pro. Read about [how to select the bandwidth for your application](../README.md#bandwidth)
description: Configures the input impedance, coupling, gain, and attenuation for each channel.
method: post
name: set_frontend
parameters:
    - default: null
      description: Target channel
      name: channel
      param_range:
          mokugo: 1, 2
          mokulab: 1, 2
          mokupro: 1, 2
          mokudelta: 1, 2
      type: integer
      unit: null
    - default: null
      description: Impedance
      name: impedance
      param_range:
          mokugo: 1MOhm
          mokulab: 50Ohm, 1MOhm
          mokupro: 50Ohm, 1MOhm
          mokudelta: 50Ohm, 1MOhm
      type: string
      unit: null
    - default: null
      description: Input Coupling
      name: coupling
      param_range: AC, DC
      type: string
      unit: null
    - default: None
      description: Input attenuation (required when gain is not set)
      name: attenuation
      param_range:
          mokugo: 0dB, 14dB
          mokulab: 0dB, 20dB
          mokupro: 0dB, 20dB, 40dB
          mokudelta: -20dB, 0dB, 20dB, 32dB
      type: string
      unit: null
    - default: None
      description: Input gain (required when attenuation is not set)
      name: gain
      param_range:
          mokugo: 0dB, -14dB
          mokulab: 0dB, -20dB
          mokupro: 0dB, -20dB, -40dB
          mokudelta: 20dB, 0dB, -20dB, -32dB
      type: string
      unit: null
    - default: 300MHz
      description: Input bandwidth
      name: bandwidth
      param_range:
          mokupro: 300MHz, 600MHz
      type: string
      unit: null
    - default: true
      description: Disable all implicit conversions and coercions.
      name: strict
      param_range: null
      type: boolean
      unit: null
summary: set_frontend
---

<headers/>
<parameters/>

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import LockInAmp
i = LockInAmp('192.168.###.###')
# Set Input 1 to 1 MOhm, DC coupled, 20 dB gain, 300 MHz bandwidth
i.set_frontend(channel=1, impedance="1MOhm", coupling="DC", attenuation="20dB", bandwidth="300MHz", strict=True)
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuLockInAmp('192.168.###.###');
% Set Input 1 to 1 MOhm, DC coupled, 20 dB gain, 300 MHz bandwidth
m.set_frontend(1, 'DC', '1MOhm', '20dB', 'bandwidth', '300MHz', 'strict', true);
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"strict": True, "channel": 1, "impedance": "1MOhm", "coupling": "DC", "attenuation": "20dB", "bandwidth": "300MHz"}'\
        http://<ip>/api/lockinamp/set_frontend
```

</code-block>

</code-group>

### Sample response

```json
{
    "bandwidth": "300MHz",
    "attenuation": "20dB",
    "coupling": "DC",
    "impedance": "1MOhm"
}
```
