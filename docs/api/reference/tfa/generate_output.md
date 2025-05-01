---
additional_doc: Outputs are matched to their corresponding Intervals. For example, Output 1 (or Output A) is always matched to Interval 1,and so on for all channels
description: Generate interval or count bases signal on the given output channel
method: post
name: generate_output
parameters:
    - default:
      description: Target channel
      name: channel
      param_range:
          mokugo: 1, 2
          mokulab: 1, 2
          mokupro: 1, 2, 3, 4
      type: integer
      unit:
    - default:
      description: Output signal type
      name: signal_type
      param_range: Interval, Count
      type: string
      unit:
    - default:
      description: Scaling ratio of the interval (in seconds) into Volts
      name: scaling
      param_range:
      type: number
      unit: V
    - default: 0
      description: Time duration that is converted to 0V
      name: zero_point
      param_range:
      type: number
      unit:
    - default: null
      description: Output range
      name: output_range
      param_range: 2Vpp, 10Vpp
      type: string
      unit:
    - default: False
      description: Whether to invert the output signal
      name: invert
      param_range:
      type: boolean
      unit:
    - default: True
      description: Disable all implicit conversions and coercions.
      name: strict
      param_range:
      type: boolean
      unit:
summary: generate_output
---

<headers/>

<parameters/>

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import TimeFrequencyAnalyzer
i = TimeFrequencyAnalyzer('192.168.###.###')
# Generate Interval output on output1 with 0 scaling and zero point
i.generate_output(channel=1, signal_type="Interval", scaling=0, zero_point=0)
# Generate Count output on output2 with 0 scaling
i.generate_output(channel=1, signal_type="Count", scaling=0)
# configure event detectors
# configure interval analyzers
# retrieve data
data = i.get_data()
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuTimeFrequencyAnalyzer('192.168.###.###')
% Generate Interval output on output1 with 0 scaling and zero point
m.generate_output(1, 'Interval', 0, 'zero_point', 0)
% Generate Count output on output2 with 0 scaling
m.generate_output(1, 'Count', 0)
% configure event detectors
% configure interval analyzers
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
    "channel" : 1,
    "signal_type":"Interval",
    "scaling":0,
    "zero_point":0
}
$: curl -H 'Moku-Client-Key: <key>' \
    -H 'Content-Type: application/json' \
    --data @request.json \
    http:// <ip >/api/tfa/generate_output
```

</code-block>

</code-group>
