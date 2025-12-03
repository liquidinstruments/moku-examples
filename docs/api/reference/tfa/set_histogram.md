---
additional_doc: The histogram is a distribution of time measurements in a series of bins. A bar’s height indicates the frequency of events detected with a time value within the corresponding timestamp-bin.
description: Sets the span of histogram
method: post
name: set_histogram
parameters:
    - default:
      description: Start time
      name: start_time
      param_range:
      type: number
      unit:
    - default:
      description: Stop time
      name: stop_time
      param_range:
      type: number
      unit:
    - default: Interval
      description: Histogram type
      name: type
      param_range: Interval, G2
      type: string
      unit:
    - default: True
      description: Disable all implicit conversions and coercions
      name: strict
      param_range:
      type: boolean
      unit:
summary: set_histogram
---

<headers/>

<parameters/>

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import TimeFrequencyAnalyzer
i = TimeFrequencyAnalyzer('192.168.###.###')
# configure histogram span to 1 second and set G2 histogram type
i.set_histogram(start_time=0, stop_time=1, type="G2")
# retrieve data
data = i.get_data()
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuTimeFrequencyAnalyzer('192.168.###.###')
% configure histogram span to 1 second and set G2 histogram type
m.set_histogram(0, 1, "G2")
% retrieve data
m.get_data()
```

</code-block>

<code-block title="cURL">

```bash
# You should create a JSON file with the data content rather than passing
# arguments on the CLI as the lookup data is necessarily very large
$: cat request.json
{
    "start_time": 0,
    "stop_time" : 1,
    "type": "G2"
}
$: curl -H 'Moku-Client-Key: <key>' \
    -H 'Content-Type: application/json' \
    --data @request.json \
    http:// <ip >/api/tfa/set_histogram
```

</code-block>

</code-group>

### Sample response

```json
{
    "HistogramType": 1,
    "start_time": 0,
    "stop_time": 1,
    "type": "g2correlationfunction"
}
```