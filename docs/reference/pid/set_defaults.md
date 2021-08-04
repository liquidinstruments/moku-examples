---
additional_doc: null
description: Set the PID Controller to sane defaults.
method: post
name: set_defaults
parameters: []
summary: set_defaults
---

<headers/>
<parameters/>

Default state implies,

- Enable output on both channels
- Set Input Coupling to DC
- Set Input Range to 10Vpp


::: tip INFO
Reference to any instrument object will always be in default state.
:::




Usage in clients, 

<code-group>
<code-block title="Python">
```python
from moku.instruments import PIDController
i = PIDController('192.168.###.###', force_connect=False)
# PIDController reference i is in default state
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuPIDController('192.168.###.###', true);
% PIDController reference m is in default state
```
</code-block>
</code-group>