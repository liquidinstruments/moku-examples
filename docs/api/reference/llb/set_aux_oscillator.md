---
additional_doc: null
description: Configures the auxiliary oscillator output signal, including amplitude, frequency, and output channel.  In the Moku app, this output is labelled “Modulation” and is commonly used for laser modulation.
method: post
name: set_aux_oscillator
parameters:
    - default: true
      description: Enable or disable auxiliary oscillator
      name: enabled
      param_range: null
      type: boolean
      unit: null
    - default: 1e6
      description: Frequency of the auxiliary oscillator
      name: frequency
      param_range:
          mokugo: 1 mHz to 20 MHz
          mokulab: 1 mHz to 250 MHz
          mokupro: 1 mHz to 500 MHz
          mokudelta: 1 mHz to 2 GHz
      type: integer
      unit: Hz
    - default: 0.5
      description: Amplitude of the auxiliary oscillator
      name: amplitude
      param_range:
          mokugo: 2 mVpp to 10 Vpp
          mokulab: 1 mVpp to 2 Vpp
          mokupro: 1 mVpp to 2 Vpp
          mokudelta: 1 mVpp to 1 Vpp
      type: integer
      unit: V
    - default: Output1
      description: Output to connect modulation signal to.
      name: output
      param_range: Output1, Output2, Output3, Output4, OutputA, OutputB, OutputC
      type: string
      unit: null
    - default: true
      description: Disable all implicit conversions and coercions.
      name: strict
      param_range: null
      type: boolean
      unit: null
summary: set_aux_oscillator
---

<headers/>

<parameters/>

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import LaserLockBox
i = LaserLockBox('192.168.###.###', force_connect=True)
# Enable the auxiliary oscillator and drive Output3 at 1 MHz with 0.5 V amplitude
i.set_aux_oscillator(enabled=True, frequency=1e6, amplitude=0.5, output='Output3')

```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuLaserLockBox('192.168.###.###', force_connect=true);
% Enable the auxiliary oscillator and route 1 MHz, 0.5 V to Output3
m.set_aux_oscillator('enabled', true, 'frequency', 1e6, 'amplitude', 0.5, ...
                     'output', 'Output3');
```

</code-block>

<code-block title="cURL">

```bash
# Enable the auxiliary oscillator and drive Output3 at 1 MHz with 0.5 V amplitude
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"enabled": true, "frequency": 1000000, "amplitude": 0.5, 
                 "output": "Output3"}'\
        http://<ip>/api/laserlockbox/set_aux_oscillator
```

</code-block>

</code-group>

### Sample response

```json
{
    "amplitude": 0.5,
    "enabled": true,
    "frequency": 1000000.0,
    "source": "AnalogOutput3"
}
```
