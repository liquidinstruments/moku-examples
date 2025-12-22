---
additional_doc: null
description: Configures the input impedance, coupling, gain, and attenuation for each channel.
method: post
name: set_frontend
parameters:
    - default: null
      description: Target channel
      name: channel
      param_range: 1, 2
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
from moku.instruments import LaserLockBox
i = LaserLockBox('192.168.###.###', force_connect=True)
# Configure channel 1 for AC coupling, 1 MΩ impedance, and 0 dB attenuation
i.set_frontend(1, "AC", "1MOhm", "0dB")
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuLaserLockBox('192.168.###.###', force_connect=true);
% Configure channel 1 for AC coupling, 1 MΩ impedance, and 0 dB attenuation
m.set_frontend(1, "AC", "1MOhm", "0dB");
```

</code-block>

<code-block title="cURL">

```bash
# Configure channel 1 for AC coupling, 1 MΩ impedance, and 0 dB attenuation
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"channel": 1, "coupling": "AC", "impedance": "1MOhm", 
                 "gain": "0dB"}'\
        http://<ip>/api/laserlockbox/set_frontend
```

</code-block>

</code-group>

### Sample response

```json
{
    "attenuation": "0dB",
    "bandwidth": "600MHz",
    "coupling": "AC",
    "impedance": "1MOhm"
}
```
