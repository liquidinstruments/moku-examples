---
additional_doc: In addition to configuring this Auxilliary sine wave output, it must be routed to an
    actual output channel using `set_outputs`
description: Configures the Auxilliary sine wave generator.
method: post
name: set_aux_output
parameters:
- default: null
  description: Sine wave frequency
  name: frequency
  type: number
  unit: Hz
- default: null
  description: Sine wave amplitude
  name: amplitude
  type: number
  unit: V
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