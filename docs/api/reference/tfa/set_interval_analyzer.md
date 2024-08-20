---
additional_doc: Intervals are used to calculate the Statistics, Data Logging values and Output values. The Intervals are determined between the input Events and an Interval is the time difference between two specific Events captured by the instrument
description: Configures start and stop events for a given numerical id of interval analyzer
method: post
name: set_interval_analyzer
parameters:
    - default:
      description: The numerical id of the interval analyzer
      name: id
      param_range:
      type: integer
      unit:
    - default:
      description: Start event
      name: start_event_id
      param_range:
      type: integer
      unit:
    - default:
      description: Stop event
      name: stop_event_id
      param_range:
      type: integer
      unit:
    - default: True
      description: Enable/disable interval analyzer
      name: enable
      param_range:
      type: boolean
      unit:
    - default: True
      description: Disable all implicit conversions and coercions.
      name: strict
      param_range:
      type: boolean
      unit:
summary: set_interval_analyzer
---

<headers/>

<parameters/>

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import TimeFrequencyAnalyzer
i = TimeFrequencyAnalyzer('192.168.###.###')
# configure interval analyzers
i.set_interval_analyzer(1, 1, 1)
# retrieve data
data = i.get_data()
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuTimeFrequencyAnalyzer('192.168.###.###')
% configure interval analyzers
m.set_interval_analyzer(1, 1, 1)
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
    "id":1,
    "start_event_id":1,
    "stop_event_id":1,
}
$: curl -H 'Moku-Client-Key: <key>' \
    -H 'Content-Type: application/json' \
    --data @request.json \
    http:// <ip >/api/tfa/set_interval_analyzer
```

</code-block>

</code-group>

### Sample response
