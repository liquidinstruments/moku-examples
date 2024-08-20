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
from moku.instruments import LaserLockBox
i = LaserLockBox('192.168.###.###')
# Following configuration produces Chebyshev type 1 IIR filter
i.set_filter(shape="Lowpass", type="ChebyshevI")
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuLaserLockBox('192.168.###.###');
% Following configuration produces Chebyshev type 1 IIR filter
m.set_filter('shape', 'Lowpass', 'type', 'ChebyshevI')
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"type":"ChebyshevI", "shape":"Lowpass"}'\
        http://<ip>/api/laserlockbox/set_filter
```

</code-block>

</code-group>

### Sample response

```json
{
    "low_pass_corner": 10000.0,
    "shape": "Lowpass"
}
```
