---
additional_doc: null
description: Set the state of the digital I/O pins
method: post
name: set_pins
parameters:

- default: null
  description: Target pin to configure
  name: pin
  param_range: Pin1, Pin2, Pin3, Pin4, Pin5, Pin6, Pin7, Pin8, Pin9, Pin10, Pin11,
    Pin12, Pin13, Pin14, Pin15, Pin16
  type: string
  unit: null
- default: null
  description: State of the target pin.
  name: state
  param_range: I, O, H, L, X
  type: string
  unit: null
- default: true
  description: Disable all implicit conversions and coercions. Please refer to [Pin Status Definitions](README.md) for the list of available statuses
  name: strict
  param_range: null
  type: boolean
  unit: null
summary: set_pin
available_on: "mokugo"
---


<headers/>
<parameters/>

Please refer to [Pin Status Definitions](README.md) for the list of available statuses

Examples,

<code-group>
<code-block title="Python">
```python{3-5}
from moku.instruments import LogicAnalyzer
i = LogicAnalyzer('192.168.###.###', force_connect=False)
i.set_pin("Pin1", "O")
i.set_pin("Pin1", "H")
i.set_pin("Pin1", "L")
i.get_pins()
```
</code-block>

<code-block title="MATLAB">
```matlab{2-4}
m = MokuLogicAnalyzer('192.168.###.###', true);
m.set_pin("Pin1", "O");
m.set_pin("Pin1", "H");
m.set_pin("Pin1", 'L');
m.get_pins();
```
</code-block>
</code-group>

:::tip Note
set_pin only configures the state of the Pin, to generate a pattern on a pin use
[generate_pattern](generate_pattern.md)
:::

