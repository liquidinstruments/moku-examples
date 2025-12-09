---
additional_doc: null
description: Configures the input impedance, coupling, gain, and bandwidth for each channel.
method: post
name: set_frontend
parameters:
    - default: null
      description: Target channel
      name: channel
      param_range:
          mokudelta: 1, 2, 3, 4
      type: integer
      unit: null
    - default: null
      description: Impedance
      name: impedance
      param_range:
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
      description: Input gain
      name: gain
      param_range:
          mokudelta: 20dB, 0dB, -20dB, -32dB
      type: string
      unit: null
    - default: 2GHz
      description: bandwidth
      name: bandwidth
      param_range:
          mokudelta: 2GHz, 1MHz
      type: string
      unit: null
    - default: true
      description: Disable all implicit conversions and coercions.
      name: strict
      param_range: null
      type: boolean
      unit: null
summary: set_frontend
available_on: 'Moku:Delta'
---

<headers/>

Only set the Gigabit Streamer+ frontend parameters when using in single-slot configuration. When used in Multi-Instrument Mode, the Gigabit Streamer instrument is no longer in control of the input attenuation/range, as that input may be shared by multiple instruments.

When in Multi-Instrument Mode, the user must use this `set_frontend` function rather than the ones "typically" found in the namespaces of individual instruments.

:::tip Bandwidth
Bandwidth is implicitly set depending on the input impedance used. See [Bandwidth](../../reference/#bandwidth) for details
:::

<parameters/>

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import GigabitStreamerPlus
m = GigabitStreamerPlus('192.168.###.###')
m.set_frontend(channel=1, coupling="DC", gain="0dB", impedance="50Ohm", strict=True)
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuGigabitStreamerPlus('192.168.###.###')
m.set_frontend(1, "50Ohm", "DC", 'gain', '0dB', 'strict': true)
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
       -H 'Content-Type: application/json'\
       --data '{"channel": 1, "coupling", "DC", "impedance": "50Ohm", "gain": "0dB", "strict": true}'\
       http://<ip>/api/gs/set_frontend
```

</code-block>

</code-group>

### Sample response

```json
{
    "bandwidth": "2GHz",
    "coupling": "DC",
    "gain": "0dB",
    "impedance": "50Ohm"
}
```
