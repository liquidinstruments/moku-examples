---
additional_doc:
    When using the Python or MATLAB clients, user can access this function directly from
    instrument reference.
description: Returns hardware version, mokuOS version, API Server version, and proxy version
method: get
name: describe
parameters: []
summary: describe
---

<headers/>

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import ArbitraryWaveformGenerator

i = ArbitraryWaveformGenerator('192.168.###.###', force_connect=False)

# Here you can access the describe function
i.describe()
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuOscilloscope('192.168.###.###', false);

% Here you can access the describe function
m.describe()

```

</code-block>

<code-block title="cURL">

```bash
$: curl http://<ip>/api/moku/describe
```

</code-block>

</code-group>

Sample response,

```json
{
    "mokuOS": "4.0.1", //MokuOS version Moku running on
    "version": "4.0.1.1", //API Server version
    "hardware": "Moku:Go",
    "proxy_version": "2",
}
```
