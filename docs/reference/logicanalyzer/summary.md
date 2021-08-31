---
additional_doc: null
description: Returns a short summary of current instrument state
method: get
name: summary
parameters: []
summary: summary
available_on: "mokugo"
---


<headers/>
<parameters/>

<code-group>
<code-block title="Python">
```python{4}
from moku.instruments import LogicAnalyzer
i = LogicAnalyzer('192.168.###.###', force_connect=False)
i.set_pin("Pin1", "O")
print(i.summary())
```
</code-block>

<code-block title="MATLAB">
```matlab{5}
m = MokuLogicAnalyzer('192.168.###.###', true);
m.set_pin("Pin1", "O");
m.set_pin("Pin1", "H");
m.set_pin("Pin1", 'L');
disp(m.summary());
```
</code-block>
</code-group>

Sample response,

```text
Moku:Go Logic Analyzer
Pin 1 - Output - Custom (62.5 MHz tick frequency, 4 values)
Pin 2 - Input
Pin 3 - Input
Pin 4 - Override High
Pin 5 - Input
Pin 6 - Input
Pin 7 - Input
Pin 8 - Input
Pin 9 - Input
Pin 10 - Input
Pin 11 - Input
Pin 12 - Output - Custom (62.5 MHz tick frequency, 2 values)
Pin 13 - Input
Pin 14 - Input
Pin 15 - Input
Pin 16 - Input
Time span 200.0 ns, time offset 0.000 s
Edge trigger: Pin 5, Auto mode, Rising edge

```