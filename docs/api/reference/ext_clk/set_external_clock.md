---
additional_doc:
    When using either of the clients, user can access this function directly from
    instrument reference.
description: Enable/Disable the 10 MHz external reference clock on the Moku 
method: post
name: set_external_clock
parameters:
    - default: true
      description: Boolean flag representing the desired state of the clock
      name: enable
      param_range: null
      type: boolean
      unit: null
summary: set_external_clock
available_on: 'Moku:Pro, Moku:Lab, Moku:Delta'
---

<headers/>

<parameters/>

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import Oscilloscope

i = Oscilloscope('192.168.###.###', force_connect=False)
# Here you can access the set_external_clock function
i.set_external_clock()
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuOscilloscope('192.168.###.###', false);

% Here you can access the set_external_clock function
m.set_external_clock()

```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{}'\
        http://<ip>/api/moku/set_external_clock
```

</code-block>

</code-group>
