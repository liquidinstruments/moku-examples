---
additional_doc: null
description: Set the Frequency Response Analyzer to default state.
method: post
name: set_defaults
parameters: []
summary: set_defaults
---

<headers/>

Default state implies,

-   Set Measurement to In/Out
-   Set Input Coupling to DC
-   Enable offset on both channels and default it to '0'

::: tip INFO
Reference to any instrument object will always be in default state.
:::

<parameters/>

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import FrequencyResponseAnalyzer
i = FrequencyResponseAnalyzer('192.168.###.###')
# FrequencyResponseAnalyzer reference i is in default state
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuFrequencyResponseAnalyzer('192.168.###.###');
% FrequencyResponseAnalyzer reference m is in default state
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{}'\
        http://<ip>/api/fra/set_defaults
```

</code-block>

</code-group>
