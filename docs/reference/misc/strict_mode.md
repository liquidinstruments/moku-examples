---
additional_doc: 
description: 
method: post
name: generate_pattern
summary: What is strict mode?
---

<headers/>

Most of the functions have an additional parameter `strict` which controls coercions of input values. 
Meaning, when `strict` is passed as `true` (which is the default) the Moku API **will not try to coerce** input values.
API returns an error with appropriate message(s), when it cannot set to what user has asked for.

Let's look at an example,

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

:::tip Note
A pattern can be generated only when the state of Pin is set to 'O'. 
Refer [Pin Status Definitions](README.md) for the list of available statuses
:::

Examples,

