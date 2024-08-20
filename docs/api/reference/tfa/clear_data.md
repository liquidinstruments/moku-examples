---
additional_doc: Clears all previously acquired data and resets acquisition period, statistics, histogram and output values
description: Clear statistics and histogram data
method: post
name: clear_data
parameters: []
summary: clear_data
---

<headers/>

<parameters/>

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import TimeFrequencyAnalyzer
i = TimeFrequencyAnalyzer('192.168.###.###')
# configure event detectors
# configure interval analyzers
# retrieve data
data = i.get_data()
# resets histogram and statistics
i.clear_data()
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuTimeFrequencyAnalyzer('192.168.###.###')
% configure event detectors
% configure interval analyzers
% retrieve data
m.get_data()
% resets histogram and statistics
m.clear_data()
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>' \
    -H 'Content-Type: application/json' \
    --data {} \
    http:// <ip >/api/tfa/set_frontend

```

</code-block>
</code-group>
