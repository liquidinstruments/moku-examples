---
additional_doc: The FIR Filter instrument has an output stage between the controller itself and the
    hardware outputs. This output stage includes the output offset and may contain a hardware gain stage
    depending on the Moku hardware platform (e.g. Moku:Pro has a selectable +14dB hardware driver required
    to obtain maximum output range).

    To have a FIR filter output present on the device, the signal must be connected to the output stage (signal=true)
    and the output stage must also be enabled (output=true).
description: Enable or disable the FIR channel output(s)
method: post
name: enable_output
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
  description: Connect/Disconnect the controller signal to/from the output stage
  name: signal
  param_range: null
  type: boolean
  unit: null
- default: null
  description: Enable/Disable output driver
  name: output
  param_range: null
  type: boolean
  unit: null
- default: "0dB"
  description: If applicable, enable the hardware's output gain stage (high output range)
  name: gain_range
  param_range: 0dB, 14dB
  type: string
  unit: dB
- default: true
  description: Disable all implicit conversions and coercions.
  name: strict
  param_range: null
  type: boolean
  unit: null
summary: enable_output
---

<headers/>
<parameters/>


### Examples

<code-group>
<code-block title="Python">
```python
from moku.instruments import FIRFilterBox
i = FIRFilterBox('192.168.###.###')
# Configure instrument to desired state

# Enable out signal and output driver
i.enable_output(1, signal=True, output=True)
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuFIRFilterBox('192.168.###.###', true);
% Configure instrument to desired state

% Enable out signal and output driver
m.enable_output(1,'signal',true,'output',true);

```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"channel": 1, "signal": true, "output": true}'\
        http://<ip>/api/firfilter/enable_output
```
</code-block>

</code-group>