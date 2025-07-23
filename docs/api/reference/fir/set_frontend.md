---
additional_doc: null
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
          mokupro: 1, 2, 3, 4
          mokudelta: 1, 2, 3, 4, 5, 6, 7, 8
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
from moku.instruments import FIRFilterBox
i = FIRFilterBox('192.168.###.###')
i.set_frontend(1, "AC", "1MOhm", "14dB")
```

</code-block>

<code-block title="MATLAB">

````matlab
m = MokuFIRFilterBox('192.168.###.###');
m.set_frontend(1, "AC", "1MOhm", "14dB");```
</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"channel": 1, "impedance": "1MOhm", "coupling": "DC", "attenuation": "14dB"}'\
        http://<ip>/api/firfilter/set_frontend
````

</code-block>

</code-group>
