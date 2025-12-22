---
additional_doc: To use customer coefficients see  `set_custom_filter`
description: Configure the Infinite Impulse Response filter with one of available filter shape and type
method: post
name: set_filter
parameters:
    - default: Lowpass
      description: IIR Filter shape
      name: shape
      type: string
      unit: null
      param_range: Lowpass, Bandstop
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
      param_range: 2, 4
      type: integer
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
from moku.instruments import LaserLockBox
i = LaserLockBox('192.168.###.###', force_connect=True)
# Configure a 4th-order Chebyshev Type I low-pass filter with a 10 kHz cutoff
i.set_filter(shape="Lowpass", type="ChebyshevI", order=4, low_corner=1e4)
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuLaserLockBox('192.168.###.###', force_connect=true);
% Configure a 4th-order Chebyshev Type I low-pass filter with a 10 kHz cutoff
m.set_filter('shape', 'Lowpass', 'type', 'ChebyshevI', 'order', 4, ...
             'low_corner', 1e4);
```

</code-block>

<code-block title="cURL">

```bash
# Configure a 4th-order Chebyshev Type I low-pass filter with a 10 kHz cutoff
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"shape": "Lowpass", "type": "ChebyshevI", "order": 4, 
                 "low_corner": 10000}'\
        http://<ip>/api/laserlockbox/set_filter
```

</code-block>

</code-group>

### Sample response

```json
{
    "Filter shape": 0,
    "low_pass_corner": 10000.0,
    "order": 4,
    "pass_band_ripple": 3.0,
    "sample_rate": "78.125 MHz",
    "shape": "Lowpass",
    "type": "ChebyshevI"
}
```
