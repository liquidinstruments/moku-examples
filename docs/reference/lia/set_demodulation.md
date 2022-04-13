---
additional_doc: null
description: Configures the demodulation source and optionally its frequency and phase
method: post
name: set_demodulation
parameters:
- default: null
  description: The demodulation source
  name: mode
  param_range: Internal, External, ExternalPLL, None
  type: string
  unit: null
- default: 1000000
  description: Frequency of internally-generated demod source
  name: frequency
  type: number
  unit: Hz
- default: 0
  description: Phase of internally-generated demod source
  name: phase
  type: number
  unit: degrees
- default: true
  description: Disable all implicit conversions and coercions.
  name: strict
  param_range: null
  type: boolean
  unit: null
summary: set_demodulation
---

<headers/>
<parameters/>

### Examples

<code-group>
<code-block title="Python">
```python
from moku.instruments import LockInAmp
i = LockInAmp('192.168.###.###')
i.set_demodulation(mode="Internal",frequency=1000000,phase=0)
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuLockInAmp('192.168.###.###');
m.set_demodulation('mode','Internal','frequency',1000000,'phase',0)
```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"mode":"Internal","frequency":1000000,"phase":0}'\
        http://<ip>/api/lockinamp/set_demodulation
```
</code-block>

</code-group>

### Sample response
```json
{
  "frequency": 1000000.0,
  "mode": "Internal",
  "phase": 0.0
}
```