---
additional_doc: null
description: Set the desired frequency span to be analyzed.
method: post
name: set_span
parameters:
    - default: null
      description: Left-most frequency
      name: frequency1
      param_range:
          mokugo: 0 to 30e6
          mokulab: 0 to 250e6
          mokupro: 0 to 300e6
          mokudelta: 0 to 2e9
      type: number
      unit: Hz
    - default: null
      description: Right-most frequency
      name: frequency2
      param_range:
          mokugo: 0 to 30e6
          mokulab: 0 to 250e6
          mokupro: 0 to 300e6
          mokudelta: 0 to 2e9
      type: number
      unit: Hz
    - default: true
      description: Disable all implicit conversions and coercions.
      name: strict
      param_range: null
      type: boolean
      unit: null
summary: set_span
---

<headers/>

Rounding and quantization
in the instrument limits the range of spans for which a full set of 1024 data points
can be calculated. This means that the resultant number of data points in SpectrumData
frames will vary with the set span. Note however that the associated frequencies
are given with the frame containing the data.

<parameters/>

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import SpectrumAnalyzer
i = SpectrumAnalyzer('192.168.###.###')
# Configure the spectrum analyzer to measure a span from 10Hz to 10MHz,
i.set_span(10, 10e6)
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuSpectrumAnalyzer('192.168.###.###');
% Configure the spectrum analyzer to measure a span from 10Hz to 10MHz,
m.set_span(10, 10e6);
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"frequency1": 10, "frequency2": 10e6}'\
        http://<ip>/api/spectrumanalyzer/set_span
```

</code-block>

</code-group>

### Sample response

```json
{
    "frequency1": 10.0,
    "frequency2": 10000000.0
}
```
