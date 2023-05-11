---
additional_doc: null
description: Configures the scan oscillator to scan or sweep the laser over a given frequency range
method: post
name: set_scan_oscillator
parameters:
- default: true
  description: Enable or disable auxiliary oscillator
  name: enabled
  param_range: null
  type: boolean
  unit: null
- default: PositiveRamp
  description: Output to connect modulation signal to. 
  name: shape
  param_range: PositiveRamp, Triangle, NegativeRamp
  type: string
  unit: null
- default: 10
  description: Frequency of the auxiliary oscillator
  name: frequency
  param_range: 1 mHz to 10 MHz
  type: integer
  unit: Hz
- default: 0.5
  description: Amplitude of the auxiliary oscillator
  name: frequency
  param_range: null
  type: integer
  unit: Hz
- default: Output1
  description: Output to connect modulation signal to. 
  name: source
  param_range: Output1, Output2, Output3, Output4, OutputA, OutputB
  type: string
  unit: null
- default: true
  description: Disable all implicit conversions and coercions.
  name: strict
  param_range: null
  type: boolean
  unit: null
summary: set_scan_oscillator
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
i.set_scan_oscillator()
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuLaserLockBox('192.168.###.###');
m.set_scan_oscillator()
```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"amplitude": 0.5, "enabled": True, "frequency": 1000000.0, "source": "Output1"}'\
        http://<ip>/api/laserlockbox/set_scan_oscillator
```
</code-block>

</code-group>

### Sample response,

```json
{
  "amplitude": 0.5,
  "enabled": true,
  "frequency": 10.0,
  "output": "Output1",
  "shape": "PositiveRamp"
}
```