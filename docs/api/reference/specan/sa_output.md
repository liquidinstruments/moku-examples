---
additional_doc: null
description: Generate a sinewave on the output channels.
method: post
name: sa_output
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
      description: The peak-to-peak amplitude of the output sinewave (On Moku:Pro, the output sine wave amplitude is limited to -1V to 1V above 100 MHz. For Moku:Delta, signals above 100 MHz are limited to ±500mV)
      name: amplitude
      param_range:
          mokugo: 1e-3 to 10
          mokulab: 1e-3 to 4
          mokupro: 1e-3 to 10
          mokudelta: 1e-3 to 10 (For Moku:Delta, signals above 100 MHz are limited to ±500mV)
      type: number
      unit: null
    - default: null
      description: Frequency of the output sinewave. (For sine waves with amplitude above 2Vpp, the maximum frequency is 100 MHz on Moku:Pro. For Moku:Delta, signals above 100 MHz are limited to ±500mV)
      name: frequency
      param_range:
          mokugo: 0 to 30e6
          mokulab: 0 to 250e6
          mokupro: 0 to 500e6
          mokudelta: 1e-3 to 2e9 (For Moku:Delta, signals above 100 MHz are limited to ±500mV)
      type: number
      unit: Hz
    - default: true
      description: Disable all implicit conversions and coercions.
      name: strict
      param_range: null
      type: boolean
      unit: null
summary: sa_output
---

<headers/>
<parameters/>

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import SpectrumAnalyzer
i = SpectrumAnalyzer('192.168.###.###')
# Generate a Sine wave on output channel 1
i.sa_output(channel=1, amplitude=0.5, frequency=1e5)
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuSpectrumAnalyzer('192.168.###.###');
% Generate a Sine wave on output channel 1
m.sa_output(1, 0.5, 1e5)
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"channel": 1, "amplitude": 0.5, "frequency": 1e5}'\
        http://<ip>/api/spectrumanalyzer/sa_output
```

</code-block>

</code-group>

### Sample response

```json
{
    "amplitude": 0.5,
    "frequency": 100000.0
}
```
