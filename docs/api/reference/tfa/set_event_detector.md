---
additional_doc: null
description: Configures the event detector for the given channel
method: post
name: set_event_detector
parameters:
    - default:
      description: The numerical id of the event
      name: id
      param_range:
      type: integer
      unit:
    - default:
      description: Event source
      name: source
      param_range:
          mokugo: InputA, InputB, Input1, Input2
          mokulab: InputA, InputB, Input1, Input2, Ext.trig.
          mokupro: InputA, InputB, Input1, Input2, Input3, Input4, Ext.trig.
          mokudelta: Input1, Input2, Input3, Input4, Input5, Input6, Input7, Input8, InputA, InputB, Ext.trig.
      type: string
      unit:
    - default: 0
      description: When the selected edge of the input signal voltage crosses a user-input threshold voltage, the device detects an event. The trigger level must be within the bounds of the signal to detect events
      name: threshold
      param_range:
      type: number
      unit: V
    - default: Rising
      description: The edge chosen for event detection.The device will look for an event on the selected edge, each time the selected edge passes through the event detectors
      name: edge
      param_range: Rising, Falling , Both
      type: string
      unit:
    - default: 0.0
      description: The duration to hold-off trigger post trigger event
      name: holdoff
      param_range: 0 to 1
      type: number
      unit: s
    - default: True
      description: Disable all implicit conversions and coercions
      name: strict
      param_range:
      type: boolean
      unit:
summary: set_event_detector
---

<headers/>

<parameters/>

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import TimeFrequencyAnalyzer
i = TimeFrequencyAnalyzer('192.168.###.###')
# Configure event detector 1
i.set_event_detector(channel=1, source="Input1", threshold=0.1, edge="Rising")
# Configure event detector 2
i.set_event_detector(channel=2, source="Input2", threshold=0.1, edge="Falling")
# retrieve data
data = i.get_data()
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuTimeFrequencyAnalyzer('192.168.###.###')
% Configure event detector 1
m.set_event_detector(1, 'Input1', 'threshold', '0.1', 'edge', 'Rising')
% Configure event detector 2
m.set_event_detector(2, 'Input2', 'threshold', '0.1', 'edge', 'Falling')
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
    "source":"Input1",
    "threshold":0.1,
    "edge":"Falling"
}
$: curl -H 'Moku-Client-Key: <key>' \
    -H 'Content-Type: application/json' \
    --data @request.json \
    http:// <ip >/api/tfa/set_event_detector
```

</code-block>

</code-group>
