---
additional_doc: null
description: Returns a short summary of current instrument state
method: get
name: summary
parameters: []
summary: summary
available_on: "Moku:Go"
---

<headers/>
<parameters/>

### Examples

<code-group>
<code-block title="Python">
```python
from moku.instruments import LogicAnalyzer
i = LogicAnalyzer('192.168.###.###')
i.set_pin("Pin1", "O")
print(i.summary())
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuLogicAnalyzer('192.168.###.###');
m.set_pin("Pin1", "O");
m.set_pin("Pin1", "H");
m.set_pin("Pin1", 'L');
disp(m.summary());
```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        http://<ip>/api/logicanalyzer/summary
```

</code-block>

</code-group>

### Sample response

```json
Moku:Go Logic Analyzer
Pin 1 - Output, Clock, 31.25 MHz
Pin 2 - Input
Pin 3 - Input
Pin 4 - Input
Pin 5 - Input
Pin 6 - Input
Pin 7 - Input
Pin 8 - Input
Pin 9 - Input
Pin 10 - Input
Pin 11 - Input
Pin 12 - Input
Pin 13 - Input
Pin 14 - Input
Pin 15 - Input
Pin 16 - Input
Time span 500.0 ms, 
time offset 250.0 ms
Edge trigger: Pin 1, 
Auto mode, 
Rising edge
```