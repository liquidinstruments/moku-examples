---
title: disable_output
additional_doc: null
description: Turn off the output channels
method: post
name: disable_output
parameters:
- default: null
  description: Target channel
  name: channel
  param_range: 1, 2, 3, 4
  type: integer
  unit: null
- default: true
  description: Disable all implicit conversions and coercions.
  name: strict
  param_range: null
  type: boolean
  unit: null
summary: disable_output
available_on: "mokupro"
---



<headers/>
<parameters/>

Examples,

<code-group>
<code-block title="Python">
```python{5,6}
from moku.instruments import MokuPhasemeter

i = MokuPhasemeter('192.168.###.###', force_connect=False)
# Disable Out 1
i.disable_output(channel=1)

```
</code-block>

<code-block title="MATLAB">
```matlab{3,4}
i = MokuPhasemeter('192.168.###.###', false);
% Disable Out 1
i.disable_output(1);

```
</code-block>
</code-group>


