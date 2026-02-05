---
additional_doc:
    In addition to configuring this Auxiliary sine wave output, it must be routed to an
    actual output channel using `set_outputs`.
description: Configures the Auxiliary sine wave generator
method: post
name: set_aux_output
parameters:
    - default: 1 MHz
      description: Sine wave frequency
      name: frequency
      param_range:
          mokugo: 1 mHz to 20 MHz
          mokulab: 1 mHz to 250 MHz
          mokupro: 1 mHz to 500 MHz
          mokudelta: 1 mHz to 2 GHz
      type: number
      unit: Hz
    - default: 500 mV
      description: Sine wave amplitude
      name: amplitude
      param_range:
          mokugo: 2 mVpp to 10 Vpp
          mokulab: 1 mVpp to 2 Vpp
          mokupro: 1 mVpp to 2 Vpp
          mokudelta: 1 mVpp to 1 Vpp
      type: number
      unit: Vpp
    - default: true
      description: Disable all implicit conversions and coercions.
      name: strict
      param_range: null
      type: boolean
      unit: null
summary: set_aux_output
---

<headers/>
<parameters/>

<code-group>
<code-block title="Python">

```python
from moku.instruments import LockInAmp
i = LockInAmp('192.168.###.###')
i.set_aux_output(frequency=1000, amplitude=1)
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuLockInAmp('192.168.###.###');
m.set_aux_output("frequency",1000,"amplitude",1)
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"frequency":1000,"amplitude":1}'\
        http://<ip>/api/lockinamp/set_aux_output
```

</code-block>

</code-group>

### Sample response

```json
{
    "amplitude": 1.0,
    "frequency": 1000.0
}
```
