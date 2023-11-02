---
additional_doc: null
description: Configures the trigger on pin(s).
method: post
name: set_trigger
parameters:
- default: null
  description: List of Pin-Trigger edge mapping
  name: pins
  param_range: Pin - 1 to 16; Trigger edge - Ignore, High, Low, Rising, Falling, Both
  type: array
  unit: null
  deprecated: true
  deprecated_text: Use sources to configure trigger sources
- default: null
  description: List of Pin-Trigger edge mapping
  name: sources
  param_range: Pin - 1 to 16; Trigger edge - Ignore, High, Low, Rising, Falling, Both
  type: array
  unit: null
- default: false
  description: Toggle advanced trigger mode
  name: advanced
  param_range: null
  type: boolean
  unit: null
  deprecated: true
- default: Auto
  description: Trigger mode
  name: mode
  param_range: Auto, Normal
  type: string
  unit: null
- default: AND
  description: Trigger combination
  name: combination
  param_range: AND, OR
  type: string
  unit: null
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
group: Oscilloscope
---


<headers/>
<parameters/>

:::tip TIP
To mimic the **Single** trigger mode, configure the trigger mode to **Normal** and call the [get_data](get_data.md) method exactly once.
:::

<code-group>
<code-block title="Python">
```python
from moku.instruments import LogicAnalyzer
i = LogicAnalyzer('192.168.###.###')
i.set_trigger([{"pin": 1, "edge": "Rising"}])
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuLogicAnalyzer('192.168.###.###');
m.set_trigger([struct('pin',1,'edge','Rising')]);
```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"pins":[{"pin":1,"edge":"Rising"}]}'\
        http://<ip>/api/logicanalyzer/set_trigger
```
</code-block>
</code-group>

### Sample response 
```json
{
  "holdoff":0.0,
  "mode":"Auto",
  "nth_event":1
}
```
