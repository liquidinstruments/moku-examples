---
additional_doc: null
description: Sets trigger source and parameters.
method: post
name: set_trigger
available_on: "mokugo"
parameters:

- default: null
  description: Trigger Source Pin
  name: source
  param_range: 1 to 16
  type: integer
  unit: null
- default: Edge
  description: Trigger type
  name: type
  param_range: Edge, Pulse
  type: string
  unit: null
- default: Auto
  description: Trigger mode
  name: mode
  param_range: Auto, Normal, Single
  type: string
  unit: null
- default: Rising
  description: Which edge to trigger on (edge mode only)
  name: edge
  param_range: Rising, Falling, Both
  type: string
  unit: null
- default: Positive
  description: Trigger pulse polarity
  name: polarity
  param_range: Positive, Negative
  type: string
  unit: null
- default: GreaterThan
  description: Trigger pulse width condition (pulse mode only)
  name: width_condition
  param_range: GreaterThan, LessThan
  type: string
  unit: null
- default: 0.0001
  description: Trigger width
  name: width
  param_range: 26e-3 to 10
  type: number
  unit: Seconds
- default: 1
  description: The number of trigger events to wait for before triggering
  name: nth_event
  param_range: 0 to 65535
  type: integer
  unit: null
- default: 0
  description: The duration to hold off Oscilloscope trigger post trigger event.
  name: holdoff
  param_range: 1e-9 to 10
  type: number
  unit: Seconds
- default: true
  description: Disable all implicit conversions and coercions.
  name: strict
  param_range: null
  type: boolean
  unit: null
summary: set_trigger
---


<headers/>
<parameters/>

<code-group>
<code-block title="Python">
```python
from moku.instruments import LogicAnalyzer
i = LogicAnalyzer('192.168.###.###')
i.set_trigger(
            source=1,         # Pin1
            type="Edge",
            mode="Auto",
            edge="Rising",
            width=0.0001,
            holdoff=0
)
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuLogicAnalyzer('192.168.###.###');
m.set_trigger(1, 'type','Edge','mode','Auto');
```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{ "type":"Edge", "source":1, "mode":"Auto", "edge":"Rising", 
"width":0.0001, "holdoff":0}'\
        http://<ip>/api/logicanalyzer/set_trigger
```
</code-block>
</code-group>

### Sample response 
```json
{
  "edge": "Rising",
  "holdoff": 0.0,
  "mode": "Auto",
  "nth_event": 1,
  "polarity": "Positive",
  "source": "Pin1",
  "type": "Edge",
  "width": 0.0001,
  "width_condition": "GreaterThan"
}
```
