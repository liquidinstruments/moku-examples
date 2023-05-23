---
additional_doc: null
description: Set FIR filter sample rate and kernel coefficients for the specified filter channel. This will enable the specified channel output.
method: post
name: set_custom_kernel_coefficients
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
  description: Desired sample rate
  name: sample_rate
  type: number
  unit: Hz
  param_range: 
    mokugo: 3.906MHz, 1.953MHz, 976.6kHz, 488.3kHz, 244.1kHz, 122.1kHz, 61.04kHz, 30.52kHz
    mokulab: 15.63MHz, 7.813MHz, 3.906MHz, 1.953MHz, 976.6kHz, 488.3kHz, 244.1kHz, 122.1kHz
    mokupro: 39.06MHz, 19.53MHz, 9.766MHz, 4.883MHz, 2.441MHz, 1.221MHz, 610.4kHz, 305.2kHz
- default: null
  description: Kernel coefficients for the specified filter channel
  name: coefficients
  param_range: null
  type: array
  unit: null
- default: true
  description: Disable all implicit conversions and coercions.
  name: strict
  param_range: null
  type: boolean
  unit: null
summary: set_custom_kernel_coefficients
---

<headers/>
<parameters/>


### Examples

<code-group>
<code-block title="Python">
```python
from moku.instruments import FIRFilterBox
i = FIRFilterBox('192.168.###.###')
# Define an array which is a simple rectangular FIR kernels with 50
# taps. A rectangular kernel produces a sinc shaped  transfer function
# with width inversely proportional to the length of the  kernel.
coefficients = [1.0 / 50.0] * 50
i.set_custom_kernel_coefficients(channel=1,
                                 sample_rate='3.906MHz',
                                 coefficients=filter_coefficients)
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuPIDController('192.168.###.###', true);
% Define an array which is a simple rectangular FIR kernels with 50
% taps. A rectangular kernel produces a sinc shaped  transfer function
% with width inversely proportional to the length of the  kernel.
coefficients = zeros(1, 40)+ (1.0 / 50.0);
m.set_custom_kernel_coefficients(1, "3.906MHz", coefficients);
```
</code-block>

<code-block title="cURL">
```bash
$: cat coefficients.json
  {
   "channel":1,
   "sample_rate":"3.906MHz",
   "coefficients": [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 
    0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02,
    0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 
    0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]
  }
 
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data @coefficients.json
        http://<ip>/api/firfilter/set_custom_kernel_coefficients
```
</code-block>

</code-group>