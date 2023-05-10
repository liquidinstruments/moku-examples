---
additional_doc: null
description: Set the infinite impulse response filter sample rate and filter coefficients for a given filter id.
method: post
name: set_custom_filter
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
- default: 1
  description: Filter output scaling
  name: scaling
  param_range: null
  type: integer
  unit: null
- default: null
  description: List of filter stages, where each stage should have six coefficients and each coefficient must be in the range [-4.0, 4.0]
  name: coefficients
  param_range: null
  type: list
  unit: null
- default: true
  description: Disable all implicit conversions and coercions.
  name: strict
  param_range: null
  type: boolean
  unit: null
summary: set_custom_filter
---

<headers/>
<parameters/>

### Examples

<code-group>
<code-block title="Python">
```python
from moku.instruments import LockInAmp

# Configure IIR filter 1 of the DFB

# The following example array produces an 8th order Direct-form 1
# Chebyshev type 2 IIR filter with a normalized stopband frequency
# of 0.2 pi rad/sample and a stopband attenuation of 40 dB.
filter_coefficients = [
    [1.0000000000, 0.6413900006, -1.0290561741, 0.6413900006, -1.6378425857, 0.8915664128],
    [1.0000000000, 0.5106751138, -0.7507394931, 0.5106751138, -1.4000444473, 0.6706551819],
    [1.0000000000, 0.3173108134, -0.3111365531, 0.3173108134, -1.0873085012, 0.4107935750]]

i.set_custom_filter(1, "3.906MHz", scaling=10,
                    coefficients=filter_coefficients)
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuLockInAmp('192.168.###.###');
% Configure IIR filter 1 of the DFB with custom coefficients

% The following example array produces an 8th order Direct-form 1
% Chebyshev type 2 IIR filter with a normalized stopband frequency
% of 0.2 pi rad/sample and a stopband attenuation of 40 dB.
filter_coefficients = [...
        1.0000000000, 0.6413900006, -1.0290561741, 0.6413900006, -1.6378425857, 0.8915664128;...
        1.0000000000, 0.5106751138, -0.7507394931, 0.5106751138, -1.4000444473, 0.6706551819;...
        1.0000000000, 0.3173108134, -0.3111365531, 0.3173108134, -1.0873085012, 0.4107935750...
    ];

m.set_custom_filter(1, '3.906MHz', 'scaling',10,'coefficients',filter_coefficients);
```
</code-block>

<code-block title="cURL">
```bash
# You should create a JSON file with the data content rather than passing
# arguments on the CLI as the request may be large
$: cat coefficients.json
{
   "channel":1,
   "sample_rate":"3.906MHz",
   "scaling":10,
   "coefficients":[
      [
         1.0000000000,
         0.6413900006,
         -1.0290561741,
         0.6413900006,
         -1.6378425857,
         0.8915664128
      ],
      [
         1.0000000000,
         0.5106751138,
         -0.7507394931,
         0.5106751138,
         -1.4000444473,
         0.6706551819
      ],
      [
         1.0000000000,
         0.3173108134,
         -0.3111365531,
         0.3173108134,
         -1.0873085012,
         0.4107935750
      ]
   ]
}
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data @coefficients.json\
        http://<ip>/api/lockinamp/set_custom_filter
```
</code-block>

</code-group>
