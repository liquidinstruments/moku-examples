---
additional_doc: null
description: Reset the Phasemeter to default state
method: post
name: set_defaults
parameters: []
summary: set_defaults
---

<headers/>

Default state implies,

-   Enable all input channels
-   Set input coupling to DC
-   Acquisition rate to 150 Hz
-   Disable all output channels

::: tip INFO
Reference to any instrument object will always be in default state.
:::

<parameters/>

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import Phasemeter
i = Phasemeter('192.168.###.###', force_connect=True)
# Phasemeter reference i is in default state
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuPhasemeter('192.168.###.###', force_connect=true);
% Phasemeter reference m is in default state
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{}'\
        http://<ip>/api/phasemeter/set_defaults
```

</code-block>

</code-group>
