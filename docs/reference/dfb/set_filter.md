---
additional_doc: To use customer coefficients see  `set_custom_filter`
description: Configure the Infinite Impulse Response filter with one of available filter shape and type
method: post
name: set_filter
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
  description: Sampling rate for the filter
  name: sample_rate
  type: number
  unit: Hz
  param_range: 
    mokugo: 3.906MHz, 488.3kHz, 61.04kHz
    mokulab: 15.625MHz, 1.9531MHz, 122.07kHz
    mokupro: 39.06MHz, 4.883MHz, 305.2kHz
- default: Lowpass
  description: IIR Filter shape
  name: shape
  type: string
  unit: null
  param_range: Lowpass, Highpass, Bandpass, Bandstop
- default: Butterworth
  description: IIR Filter type
  name: type
  type: string
  unit: null
  param_range: Butterworth, ChebyshevI, ChebyshevII, Elliptic, Cascaded, Bessel, Gaussian, Legendre
- default: undefined
  description: Low corner frequency
  name: low_corner
  param_range: null
  type: number
  unit: Hz
- default: undefined
  description: High corner frequency
  name: high_corner
  param_range: null
  type: number
  unit: Hz
- default: undefined
  description: Pass band ripple
  name: pass_band_ripple
  param_range: null
  type: number
  unit: dB
- default: undefined
  description: Stopband attenuation
  name: stop_band_attenuation
  param_range: null
  type: number
  unit: dB
- default: null
  description: Number of coefficients for the filter
  name: order
  param_range: null
  type: number
  unit: null
- default: true
  description: Disable all implicit conversions and coercions.
  name: strict
  param_range: null
  type: boolean
  unit: null
summary: set_filter
---

<headers/>
<parameters/>


### Examples

<code-group>
<code-block title="Python">
```python
from moku.instruments import DigitalFilterBox
i = DigitalFilterBox('192.168.###.###')
# Following configuration produces Chebyshev type 1 IIR filter
i.set_filter(1, "3.906MHz", shape="Lowpass", type="ChebyshevI")
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuDigitalFilterBox('192.168.###.###');
% Following configuration produces Chebyshev type 1 IIR filter
m.set_filter(1, '3.906MHz', 'shape', 'Lowpass', 'type', 'ChebyshevI')
```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"channel": 1, "sample_rate":"3.906MHz", "type":"ChebyshevI", "shape":"Lowpass"}'\
        http://<ip>/api/digitalfilterbox/set_filter
```
</code-block>

</code-group>

### Sample response
```json
{
  "low_pass_corner":10000.0,
  "sample_rate":"3.9063 MHz",
  "shape":"Lowpass"
}
```