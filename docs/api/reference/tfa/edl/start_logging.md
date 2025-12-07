---
additional_doc: null
description: Start logging configured intervals
method: post
name: start_logging
parameters:
    - default:
      description: List of event IDs to log (starting from 1)
      name: event_ids
      param_range:
      type: array
      unit:
    - default: 60
      description: Duration to log for
      name: duration
      param_range:
      type: number
      unit:
    - default: ''
      description: Optional file name prefix
      name: file_name_prefix
      param_range:
      type: string
      unit:
    - default: ''
      description: Optional comments to be included
      name: comments
      param_range:
      type: string
      unit:
    - default: 0
      description: Delay the start by
      name: delay
      param_range:
      type: integer
      unit:
    - default: EventTimestamp
      description: The quantity to log (currently only event timestamp is supported)
      name: quantity
      param_range: EventTimestamp
      type: string
      unit:
    - default: True
      description: Disable all implicit conversions and coercions
      name: strict
      param_range:
      type: boolean
      unit:
summary: start_logging
---

<headers/>

<parameters/>

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import TimeFrequencyAnalyzer
i = TimeFrequencyAnalyzer('192.168.###.###')
# log events 1 and 2 for 30 seconds with a file name prefix
i.start_logging(event_ids=[1, 2], duration=30, file_name_prefix='run1')
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuTimeFrequencyAnalyzer('192.168.###.###');
% log event 1 for 60 seconds with comments
m.start_logging([1,2], 'duration', 60, 'comments', 'run1');
```

</code-block>

<code-block title="cURL">

```bash
$ curl -H 'Moku-Client-Key: <key>' \
    -H 'Content-Type: application/json' \
    --data '{"duration": 60, "comments": "run1"}'\
    http://<ip>/api/tfa/start_logging
```

</code-block>
</code-group>

### Sample response

```json
{
  "Save to": 2, 
  "Start": 0, 
  "comments": '', 
  "duration": 0.1, 
  "event_indices": [1, 2], 
  "file_name": "run1_20251105_142702.li", 
  "file_name_prefix": "run1", 
  "location": "ssd"
}
```
