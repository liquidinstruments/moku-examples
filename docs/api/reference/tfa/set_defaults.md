---
additional_doc: null
description: Reset the Time Interval Analyzer to default state
method: post
name: set_defaults
parameters: []
summary: set_defaults
---

<headers/>

<parameters/>

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import TimeFrequencyAnalyzer
i = TimeFrequencyAnalyzer('192.168.###.###')
i.set_defaults()

```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuTimeFrequencyAnalyzer('192.168.###.###')
m.set_defaults()
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>' \
        -H 'Content-Type: application/json \
        -d '{}'
http:// <ip >/api/tfa/set_defaults
```

</code-block>

</code-group>

### Sample response
