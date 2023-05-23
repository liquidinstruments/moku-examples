---
additional_doc: null
description: Set the the FIR filter to its default state.
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



::: tip INFO
Reference to any instrument object will always be in default state.
:::




### Examples

<code-group>
<code-block title="Python">
```python
from moku.instruments import FIRFilterBox
i = FIRFilterBox('192.168.###.###')
i.set_defaults()
# PIDController reference i is in default state
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuFIRFilterBox('192.168.###.###', true);
m.set_defaults();
% PIDController reference m is in default state
```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{}'\
        http://<ip>/api/firfilter/set_defaults
```
</code-block>

</code-group>