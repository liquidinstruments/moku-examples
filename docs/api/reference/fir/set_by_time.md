---
additional_doc: null
description: Configure the FIR filter impulse response.
method: post
name: set_by_time
parameters:
    - default: null
      description: Target channel
      name: channel
      param_range:
          mokugo: 1, 2
          mokulab: 1, 2
          mokupro: 1, 2, 3, 4
          mokudelta: 1, 2, 3, 4
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
    - default: Sinc
      description: Impulse response shape
      name: response
      param_range: Rectangular, Sinc, Triangular, Gaussian
      type: string
      unit: null
    - default: undefined
      description: Impulse response width
      name: response_width
      param_range: null
      type: integer
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
summary: set_by_time
---

<headers/>
<parameters/>

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import FIRFilterBox
i = FIRFilterBox('192.168.###.###')
# Configure time domain with Sinc as impulse response
i.set_by_time(channel=1, sample_rate='3.906MHz', response="SinC",
                  response_width=10)
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuPIDController('192.168.###.###', true);
% Configure time domain with Sinc as impulse response
m.set_by_time(1, '3.906MHz', 'response', 'SinC', 'response_width', 10);
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"coefficient_count":201,"response":"Sinc","response_width":10.0,"sample_rate":"3.906MHz"}'\
        http://<ip>/api/pid/set_by_time
```

</code-block>

</code-group>

### Sample response

```json
{
    "coefficient_count": 201,
    "kaiser_order": 7,
    "response": "Sinc",
    "response_width": 10.0,
    "sample_rate": "3.906 MHz",
    "window": "None",
    "window_width": 50.0
}
```
