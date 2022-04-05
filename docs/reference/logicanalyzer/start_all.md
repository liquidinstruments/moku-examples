---
additional_doc: null
description: Starts pattern generation for all pins configured to support it
method: post
name: start_all
parameters: []
summary: start_all
available_on: "mokugo"
---

<headers/>
<parameters/>

:::tip Note
start_all should only be called after configuring the pin, meaning, set the state of the pin 
(refer [set_pin](set_pins.md)) and generating a pattern (refer [generate_pattern](generate_pattern.md))

Generating a pattern after **start_all()** does not reflect unless called again
:::

### Examples


<code-group>

<code-block title="Python">
```python
from moku.instruments import LogicAnalyzer
i = LogicAnalyzer('192.168.###.###')
i.set_pin(1, "O")
i.set_pin(2, "H")
i.set_pin(3, "L")
# Configure the output pattern for Pin 1
i.generate_pattern(pin=1, pattern=[1, 0, 0, 0, 0, 0, 0, 0])
i.start_all()
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuLogicAnalyzer('192.168.###.###');
m.set_pin('pin',1, 'state',"O");
m.set_pin('pin',2, 'state', "H");
m.set_pin('pin',3,'state', "L");
% Configure the output pattern on Pin 1 to [1 1 0 0]
m.generate_pattern('pin',1,'pattern', [1 1 0 0]);
m.start_all()
```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{}'\
        http://<ip>/api/logicanalyzer/start_all
```
</code-block>
</code-group>