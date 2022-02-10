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
```python{8}
from moku.instruments import LogicAnalyzer
i = LogicAnalyzer('192.168.###.###', force_connect=False)
i.set_pin("Pin1", "O")
i.set_pin("Pin1", "H")
i.set_pin("Pin1", "L")
# Configure the output pattern for Pin 1
i.generate_pattern("Pin1", [1, 0, 0, 0, 0, 0, 0, 0])
i.start_all()
```
</code-block>

<code-block title="MATLAB">
```matlab{7}
m = MokuLogicAnalyzer('192.168.###.###', true);
m.set_pin("Pin1", "O");
m.set_pin("Pin1", "H");
m.set_pin("Pin1", 'L');
% Configure the output pattern on Pin 8 to [1 1 0 0]
m.generate_pattern('Pin1', [1 1 0 0]);
m.start_all()
```
</code-block>
</code-group>