---
additional_doc: null
description: Configures the demodulation source and optionally its frequency and phase
method: post
name: set_demodulation
parameters:
- default: null
  description: The demodulation source
  name: mode
  param_range: Modulation, Internal, External, ExternalPLL, None
  type: string
  unit: null
- default: 1000000
  description: Frequency of internally-generated demod source
  name: frequency
  param_range:
   mokupro: 1 mHz to 600 MHz
   mokulab: 1 mHz to 200 MHz
   mokugo: 1 mHz to 20 MHz
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
mark_as_beta: true
---

<headers/>
<parameters/>

### Examples

<code-group>
<code-block title="Python">
```python
from moku.instruments import LaserLockBox
i = LaserLockBox('192.168.###.###')
i.set_demodulation(mode="Internal",frequency=1000000,phase=0)
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuLaserLockBox('192.168.###.###');
m.set_demodulation('mode','Internal','frequency',1000000,'phase',0)
```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"mode":"Internal","frequency":1000000,"phase":0}'\
        http://<ip>/api/laserlockbox/set_demodulation
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