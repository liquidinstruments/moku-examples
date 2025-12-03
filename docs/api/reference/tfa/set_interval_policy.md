---
additional_doc: For all acquisition types determine how multiple start events are handled when measuring intervals and for gated mode, select whether to discard or close the interval at the end of the gate period. 
description: Sets how corner interval cases are handled
method: post
name: set_interval_policy
parameters:
    - default: null
      description: Measure interval from first or last start event when multiple start events are detected
      name: multiple_start_events
      param_range: Use First, Use Last
      type: string
      unit: null
    - default: null
      description: Discard or close incomplete intervals when gate period ends
      name: incomplete_intervals
      param_range: Discard, Close
      type: string
      unit: null
    - default: True
      description: Disable all implicit conversions and coercions
      name: strict
      param_range: null
      type: boolean
      unit: null
summary: set_interval_policy
---

<headers/>

<parameters/>

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import TimeFrequencyAnalyzer
i = TimeFrequencyAnalyzer('192.168.###.###')
# configure interval policy to use the first start event and for the iterval to close at the end 
of the gate period
i.set_interval_policy(multiple_start_events='Use first', incomplete_intervals='Close')
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuTimeFrequencyAnalyzer('192.168.###.###')
% configure interval policy to use the first start event and for the iterval to close at the end of the gate period
m.set_interval_policy('Use first', 'Close')
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>' \
    -H 'Content-Type: application/json' \
    --data '{"multiple_start_events": "Use first", "incomplete_intervals": "Close"}' \
    http://<ip>/api/tfa/set_interval_policy
```

</code-block>

</code-group>

### Sample response

```json
{
  "multiple_start_events": "Usefirst",
  "incomplete_intervals": "Close"
}
```
