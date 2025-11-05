---
additional_doc: null
description: Sets how multiple start events are handled when measuring intervals
method: post
name: set_interval_policy
parameters:
    - default: null
      description: Measure interval from first or last start event when multiple start events are detected
      name: multiple_start_events
      param_range: Use first, Use last
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
# configure interval policy to use the first start event
i.set_interval_policy(multiple_start_events='Use first')
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuTimeFrequencyAnalyzer('192.168.###.###')
% configure interval policy to use the last start event
m.set_interval_policy('Use first')
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>' \
    -H 'Content-Type: application/json' \
    --data '{"multiple_start_events": "Use first"}' \
    http://<ip>/api/tfa/set_interval_policy
```

</code-block>

</code-group>

### Sample response

```plaintext
{
  'multiple_start_events': 'Usefirst'
}
```
