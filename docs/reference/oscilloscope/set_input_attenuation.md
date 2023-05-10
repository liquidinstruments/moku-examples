---
additional_doc: null
description: Configures the signal source of each channel
method: post
name: set_input_attenuation
parameters:
- default: null
  description: Target channel
  name: channel
  param_range:
   mokugo: 1, 2
   mokulab: 1, 2
   mokupro: 1, 2, 3, 4
  type: integer
  unit: null
- default: null
  description: Set input probe attenuation
  name: attenuation
  param_range: 1, 10000
  type: integer
  unit: null
- default: true
  description: Disable all implicit conversions and coercions.
  name: strict
  param_range: null
  type: boolean
  unit: null
summary: set_input_attenuation
---

<headers/>
<parameters/>

<code-group>
<code-block title="Python">
```python
from moku.instruments import Oscilloscope
# Configure the instrument
i = Oscilloscope('192.168.###.###')
# Set the data source of Channel 1 to be Input 1
i.set_source(1,'Input1')
# Set the input attenuation to 10x
i.set_input_attenuation(1, 10)
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuOscilloscope('192.168.###.###', true);
%% Configure the instrument
m.generate_waveform(1, 'Sine', 'amplitude',0.5, 'frequency', 10e3);
m.generate_waveform(2, 'Square', 'amplitude',1, 'frequency',20e3, 'duty', 50);

% Set the data source of Channel 1 to be Input 1
m.set_source(1,'Input1');
% Set the input attenuation to 10x
m.set_input_attenuation(1, 10);
```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"channel": 1, "attenuation": 10}'\
        http://<ip>/api/oscilloscope/set_input_attenuation
```
</code-block>

</code-group>

### Sample response
```json
{
  "attenuation":10.0
}
```