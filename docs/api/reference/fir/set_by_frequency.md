---
additional_doc: null
description: Configure the FIR filter frequency response.
method: post
name: set_by_frequency
parameters:
    - default: null
      description: Target channel
      name: channel
      param_range:
          mokugo: 1, 2
          mokulab: 1, 2
          mokupro: 1, 2, 3, 4
          mokudelta: 1, 2, 3, 4, 5, 6, 7, 8
      type: integer
      unit: null
    - default: null
      description: Desired sample rate
      name: sample_rate
      type: number
      unit: Hz
      param_range:
          mokugo: 3.906MHz, 1.953MHz, 976.6kHz, 488.3kHz, 244.1kHz, 122.1kHz, 61.04kHz, 30.52kHz
          mokulab: 15.63MHz, 7.813MHz, 3.906MHz, 1.953MHz, 976.6kHz, 488.3kHz, 244.1kHz, 122.1kHz
          mokupro: 39.06MHz, 19.53MHz, 9.766MHz, 4.883MHz, 2.441MHz, 1.221MHz, 610.4kHz, 305.2kHz
          mokudelta: 39.06MHz, 19.53MHz, 9.766MHz, 4.883MHz, 2.441MHz, 1.221MHz, 610.4kHz, 305.2kHz
    - default: 201
      description: Coefficient or tap count
      name: coefficient_count
      param_range: null
      type: number
      unit: null
    - default: Lowpass
      description: Filter shape
      name: shape
      param_range: Lowpass, Highpass, Bandpass, Bandstop
      type: string
      unit: null
    - default: undefined
      description: Low corner frequency. The low_corner is expressed as a fraction of the sample rate. For instance, if the sample rate is 3.906 MHz and the low_corner value is 0.1, the resulting low corner frequency is 390.6 kHz.
      name: low_corner
      param_range: 0.0001 to 0.49
      type: number
      unit: null
    - default: undefined
      description: High corner frequency. The high_corner is expressed as a fraction of the sample rate. For instance, if the sample rate is 3.906 MHz and the high_corner value is 0.1, the resulting high corner frequency is 390.6 kHz.
      name: high_corner
      param_range: 0.0001 to 0.49
      type: number
      unit: null
    - default: None
      description: Window function
      name: window
      param_range: None, Bartlett, Hann, Hamming, Blackman, Nuttall, Tukey, Kaiser
      type: string
      unit: null
    - default: 50
      description: Window width (Only when window function is Tukey)
      name: window_width
      param_range: null
      type: integer
      unit: null
    - default: 7
      description: Beta/Order for kaiser window (Only when window function is Kaiser)
      name: kaiser_order
      param_range: null
      type: integer
      unit: null
    - default: true
      description: Disable all implicit conversions and coercions.
      name: strict
      param_range: null
      type: boolean
      unit: null
summary: set_by_frequency
---

<headers/>
<parameters/>

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import FIRFilterBox
i = FIRFilterBox('192.168.###.###', force_connect=False)
# Configure frequency domain with response as Lowpass filter on FIR filter 1
i.set_by_frequency(1, "3.906MHz", coefficient_count=201,
                   shape="Lowpass")
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuFIRFilterBox('192.168.###.###', true);
% Configure frequency domain with response as Lowpass filter on FIR filter 1
i.set_by_frequency(1, "3.906MHz", 'coefficient_count', 201, 'shape', 'Lowpass');
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"channel":1, "sample_rate":"3.906MHz", "coefficient_count":201, "shape":"Lowpass" }'\
        http://<ip>/api/firfilter/set_by_frequency
```

</code-block>

</code-group>

### Sample response

```json
{
    "coefficient_count": 201,
    "kaiser_order": 7,
    "low_corner": 0.1,
    "sample_rate": "3.906 MHz",
    "shape": "Lowpass",
    "window": "None",
    "window_width": 50.0
}
```
