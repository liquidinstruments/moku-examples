---
additional_doc: null
description: Returns a short summary of current instrument state
method: get
name: summary
parameters: []
summary: summary
---

<headers/>
<parameters/>

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import LogicAnalyzer
i = LogicAnalyzer('192.168.###.###')
i.set_pin_mode(1, "PG1")
print(i.summary())
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuLogicAnalyzer('192.168.###.###');
m.set_pin_mode(1, "PG1");
disp(m.summary())
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
Pin 1 - Bit 0 - Output 1
Pin 2 - Bit 1 - Input
Pin 3 - Bit 2 - Input
Pin 4 - Bit 3 - Input
Pin 5 - Bit 4 - Input
Pin 6 - Bit 5 - Input
Pin 7 - Bit 6 - Input
Pin 8 - Bit 7 - Input
Pin 9 - Bit 8 - Input
Pin 10 - Bit 9 - Input
Pin 11 - Bit 10 - Input
Pin 12 - Bit 11 - Input
Pin 13 - Bit 12 - Input
Pin 14 - Bit 13 - Input
Pin 15 - Bit 14 - Input
Pin 16 - Bit 15 - Input
Time span 500.0 ms, time offset 250.0 ms
Basic trigger, Auto mode, Pin 1, Rising edge
Pattern Generator 1: 8 values, divider 125, baud rate 1,000,000
Pattern Generator 2: 8 values, divider 125, baud rate 1,000,000
```
