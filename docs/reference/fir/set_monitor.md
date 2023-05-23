---
additional_doc: null
description: Configures the specified monitor channel to view the desired filter signal.
method: post
name: set_monitor
parameters:
- default: null
  description: Monitor channel
  name: monitor_channel
  param_range:
   mokugo: 1, 2
   mokulab: 1, 2
   mokupro: 1, 2, 3, 4
  type: integer
  unit: null
- default: null
  description: Monitor channel source. 
  name: source
  param_range: 
    mokugo: None, Input1, Filter1, Output1, Input2, Filter2, Output2
    mokulab: None, Input1, Filter1, Output1, Input2, Filter2, Output2
    mokupro: None, Input1, Filter1, Output1, Input2, Filter2, Output2, Input3, Filter3, Output3, Input4, Filter4, Output4
  type: string
  unit: null
- default: true
  description: Disable all implicit conversions and coercions.
  name: strict
  param_range: null
  type: boolean
  unit: null
summary: set_monitor
group: Oscilloscope
---

<headers/>

There are two monitoring channels available, each of these can be assigned
to source signals from any of the internal FIR instrument monitoring points. Signals
larger than 12-bits must be either truncated or clipped to the allowed size.

Source signal can be one of,
 - Input1 : Channel 1 ADC input
 - Filter1 : FIR Channel 1 input (after mixing, offset and scaling)
 - Output1 : FIR Channel 1 output
 - Input2 : Channel 2 ADC Input
 - Control2 : FIR Channel 2 input (after mixing, offset and scaling)
 - Output2 : FIR Channel 2 output

<parameters/>

### Examples

<code-group>
<code-block title="Python">
```python
from moku.instruments import FIRFilterBox
i = FIRFilterBox('192.168.###.###')
# Configure instrument to desired state
i.set_monitor(1, 'Output1')
i.set_monitor(2, 'Output2')

```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuFIRFilterBox('192.168.###.###');
% Configure instrument to desired state
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
        http://<ip>/api/firfilter/set_monitor
```
</code-block>

</code-group>

### Sample response,

```json
{
  "source": "Output1"
}
```