---
additional_doc: null
description: Configures the signal source of each channel
method: post
name: set_interpolation
parameters:
- default: SinX
  description: Set interploation mode
  name: interpolation
  param_range: Linear, SinX, Gaussian
  type: string
  unit: null
- default: true
  description: Disable all implicit conversions and coercions.
  name: strict
  param_range: null
  type: boolean
  unit: null
summary: set_interpolation
---

<headers/>
<parameters/>

<code-group>
<code-block title="Python">
```python
from moku.instruments import Oscilloscope
# Configure the instrument
i = Oscilloscope('192.168.###.###')
i.generate_waveform(1, 'Sine', amplitude=0.5, frequency=10e3)
i.generate_waveform(2, 'Square', amplitude=1, frequency=20e3)

# Set the data source of Channel 1 to be Input 1
i.set_source(1,'Input1')
# Set linear interpolation
i.set_interpolation('Linear')
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
% Set linear interpolation
m.set_interpolation('Linear')
```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"interpolation": "Linear"}'\
        http://<ip>/api/oscilloscope/set_interpolation
```
</code-block>

</code-group>

### Sample response
```json
{
  "interpolation":"Linear"
}
```