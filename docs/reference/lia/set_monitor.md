---
additional_doc: null
description: Configures the specified monitor channel to view the desired PID instrument
  signal.
method: post
name: set_monitor
parameters:
- default: true
  description: Disable all implicit conversions and coercions.
  name: strict
  param_range: null
  type: boolean
  unit: null
- default: null
  description: Monitor channel
  name: monitor_channel
  param_range: null
  type: integer
  unit: null
- default: null
  description: Monitor channel source. 
  name: source
  param_range: None, Input1, Input2, ISignal, QSignal, MainOutput, AuxOutput, Demod
  type: string
  unit: null
summary: set_monitor
group: Monitors
---

<headers/>

There are two monitoring channels available, each of these can be assigned
to source signals from any of the internal LIA instrument monitoring points.

Source signal can be one of,
 - Input1 : Channel 1 ADC input
 - Input2 : Channel 2 ADC Input
 - ISignal : Quadrature mixer in-phase ("I") output
 - QSignal : Quadrature mixer quadrature ("Q") output
 - MainOutput : LIA Main output signal, see `set_outputs`
 - AuxOutput : Auxiliary output signal, see `set_outputs`
 - Demod : Signal currently being used for demodulation, see `set_demodulation`

<parameters/>

### Examples

<code-group>
<code-block title="Python">
```python
from moku.instruments import LockInAmp
i = LockInAmp('192.168.###.###')
# Set the probes to monitor Input 1 and Input 2
i.set_monitor(1, 'Input1')
i.set_monitor(2, 'Input2')

```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuLockInAmp('192.168.###.###');
% Set the probes to monitor Input 1 and Input 2
m.set_monitor(1, 'Input1')
m.set_monitor(2, 'Input2')
```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"monitor_channel": 1, "source": "Input1"}'\
        http://<ip>/api/lockinamp/set_monitor
```
</code-block>

</code-group>

### Sample response,
```json
{
  "source": "Input1"
}
```