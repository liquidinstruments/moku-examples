---
additional_doc: null
description: Generate pattern on a single channel
method: post
name: generate_pattern
parameters:
- default: null
  description: Pin to generate pattern on
  name: pin
  param_range: Pin1, Pin2, Pin3, Pin4, Pin5, Pin6, Pin7, Pin8, Pin9, Pin10, Pin11,
    Pin12, Pin13, Pin14, Pin15, Pin16
  type: string
  unit: null
- default: null
  description: Pattern to generate, array filled with 0's and 1's. Maximum size is 1024
  name: pattern
  param_range: null
  type: array
  unit: null
- default: 1
  description: Divider to scale down the base frequency of 125 MHz to the tick frequency.
    For example, a divider of 2 provides a 62.5 MHz tick frequency.
  name: divider
  param_range: 1 to 1e6
  type: integer
  unit: null
- default: true
  description: Repeat forever
  name: repeat
  param_range: null
  type: boolean
  unit: null
- default: 1
  description: Number of iterations, valid when repeat is set to false
  name: iterations
  param_range: 1 to 8192
  type: integer
  unit: null
- default: true
  description: Disable all implicit conversions and coercions.
  name: strict
  param_range: null
  type: boolean
  unit: null
summary: generate_pattern
---

<headers/>
<parameters/>

:::tip Note
A pattern can be generated only when the state of Pin is set to 'O'. 
Refer [Pin Status Definitions](README.md) for the list of available statuses
:::

Examples,

<code-group>
<code-block title="Python">
```python{6,7}
from moku.instruments import LogicAnalyzer
i = LogicAnalyzer('192.168.###.###', force_connect=False)
i.set_pin("Pin1", "O")
i.set_pin("Pin1", "H")
i.set_pin("Pin1", "L")
# Configure the output pattern for Pin 1
i.generate_pattern("Pin1", [1, 0, 0, 0, 0, 0, 0, 0])
```
</code-block>

<code-block title="MATLAB">
```matlab{5,6}
m = MokuLogicAnalyzer('192.168.###.###', true);
m.set_pin("Pin1", "O");
m.set_pin("Pin1", "H");
m.set_pin("Pin1", 'L');
% Configure the output pattern on Pin 8 to [1 1 0 0]
m.generate_pattern('Pin1', [1 1 0 0]);
```
</code-block>
</code-group>