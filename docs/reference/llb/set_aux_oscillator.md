---
additional_doc: null
description: Configures the auxiliary oscillator to the desired output channel
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
  param_range: 1 mHz to 300 MHz
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
summary: set_aux_oscillator

available_on: "Moku:Pro"
---

<headers/>

<parameters/>

### Examples

<code-group>
<code-block title="Python">
```python
from moku.instruments import LaserLockBox
i = LaserLockBox('192.168.###.###')
i.set_aux_oscillator()
# Set the probes to monitor Output 1 and Output 2
i.set_monitor(1, 'Output1')
i.set_monitor(2, 'Output2')

```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuLaserLockBox('192.168.###.###');
m.set_aux_oscillator()
% Set the probes to monitor Output 1 and Output 2
m.set_monitor(1, 'Output1')
m.set_monitor(2, 'Output2')
```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"channel": 1, "source": "Output1"}'\
        http://<ip>/api/laserlockbox/set_monitor
```
</code-block>

</code-group>

### Sample response,

```json
{
  "source": "Output1"
}
```