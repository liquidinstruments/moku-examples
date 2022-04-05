---
additional_doc: null
description: Sets up commonly used configurations in the Spectrum Analyzer, including
  the frequency range, channel signal source, resolution bandwidth, and window function.
method: post
name: sa_measurement
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
  description: Left-most frequency
  name: frequency1
  param_range:
   mokugo: 0 to 30e6
   mokupro: 0 to 300e6 
  type: number
  unit: Hz
- default: null
  description: Right-most frequency
  name: frequency2
  param_range:
   mokugo: 0 to 30e6
   mokupro: 0 to 300e6
  type: number
  unit: Hz
- default: Auto
  description: Desired resolution bandwidth (Hz)
  name: rbw
  param_range: Auto, Manual, Minimum
  type: string
  unit: null
- default: 5000
  description: RBW value (only in manual mode)
  name: rbw_value
  param_range: null
  type: number
  unit: null
- default: BlackmanHarris
  description: Window Function
      to BlackmanHarris)'
  name: window
  param_range: BlackmanHarris, FlatTop, Hanning, Rectangular
  type: string
  unit: null
- default: true
  description: Disable all implicit conversions and coercions.
  name: strict
  param_range: null
  type: boolean
  unit: null
summary: sa_measurement
---




<headers/>
<parameters/>


### Examples

<code-group>
<code-block title="Python">
```python
from moku.instruments import SpectrumAnalyzer
i = SpectrumAnalyzer('192.168.###.###')
# Configure the instrument
# Configure the spectrum analyzer to measure a span from 10Hz to 10MHz,
# auto mode, BlackmanHarris window, and no video filter
i.sa_measurement(channel=1, frequency1=10, frequency2=10e6)
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuSpectrumAnalyzer('192.168.###.###');
%% Configure the instrument
% Configure the spectrum analyzer to measure a span from 10Hz to 10MHz,
% auto mode, BlackmanHarris window, and no video filter
m.sa_measurement(1, 10, 10e6)
```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"channel": 1, "frequency1": 10, "frequency2": 10e6}'\
        http://<ip>/api/spectrumanalyzer/sa_measurement
```
</code-block>


</code-group>

### Sample response
```json
{
  "frequency1": 10.0,
  "frequency2": 10000000.0,
  "rbw": "Auto",
  "window": "BlackmanHarris"
}
```