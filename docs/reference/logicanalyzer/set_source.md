---
additional_doc: Mokus' analog inputs can be utilized as digital signal pins by configuring a threshold voltage. 
  When the signal exceeds the threshold voltage, it is detected as "High"; otherwise, it is considered "Low."
description: Sets the source of Logic Analyzer's input to either digital I/O or analog inputs
method: post
name: set_source
parameters:
- default: null
  description: Type of input source to configure
  name: source
  param_range: DigitalIO, AnalogInputs, SlotInput
  type: string
  unit: null
- default: true
  description: Disable all implicit conversions and coercions.
  name: strict
  param_range: null
  type: boolean
  unit: null
summary: set_source
---

<headers/>
<parameters/>

### Examples

<code-group>
<code-block title="Python">
```python
from moku.instruments import LogicAnalyzer
i = LogicAnalyzer('192.168.###.###')
i.set_source(source="DigitalIO")
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuLogicAnalyzer('192.168.###.###');
m.set_source('source', 'DigitalIO');
```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"source":"DigitalIO"}'\
        http://<ip>/api/logicanalyzer/set_source
```
</code-block>

</code-group>

### Sample response
```json
{"source":"DigitalIO"}
```
