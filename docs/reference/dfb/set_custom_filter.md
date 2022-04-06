---
additional_doc: null
description: Configures the LIA low-pass filter
method: post
name: set_custom_filter
parameters:
- default: null
  description: Target channel
  name: channel
  param_range:
   mokugo: 1, 2
   mokupro: 1, 2, 3, 4
  type: integer
  unit: null
- default: null
  description: Sampling rate for the filter
  name: sample_rate
  type: number
  unit: Hz
  param_range: 
    mokugo: 3.906MHz, 488.3kHz, 61.04kHz
    mokupro: 39.06MHz, 4.883MHz, 305.2kHz
- default: 1
  description: Filter output scaling
  name: scaling
  param_range: null
  type: integer
  unit: null
- default: null
  description: List of filter stages, where each stage should have six coefficients and each coefficient must be in the range [-4.0, 4.0]
  name: coefficients
  param_range: null
  type: list
  unit: null
- default: true
  description: Disable all implicit conversions and coercions.
  name: strict
  param_range: null
  type: boolean
  unit: null
summary: set_custom_filter
---

<headers/>
<parameters/>

### Examples

<code-group>
<code-block title="Python">
```python
from moku.instruments import LockInAmp
i = LockInAmp('192.168.###.###')
i.set_filter(corner_frequency=100,slope="Slope6dB")
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuLockInAmp('192.168.###.###');
m.set_filter('corner_frequency',100,'slope','Slope6dB')
```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"corner_frequency":100,"slope":"Slope6dB"}'\
        http://<ip>/api/lockinamp/set_filter
```
</code-block>

</code-group>

### Sample response
```json
{
  "corner_frequency": 99.9774868744271,
  "slope": "Slope6dB"
}
```