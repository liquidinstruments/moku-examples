---
additional_doc: Disabling the cross correlation functionality will prevent the get_data method from returning the xcorr data element.
description: Disable cross correlation.
method: post
name: disable_xcorr
parameters: []
summary: disable_xcorr
---

<headers/>

<parameters/>

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import SpectrumAnalyzer
i = SpectrumAnalyzer("192.168.###.###")
i.disable_xcorr()
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuSpectrumAnalyzer("192.168.###.###")
m.disable_xcorr()
```

</code-block>

<code-block title="cURL">

```bash
# You should create a JSON file with the data content rather than passing
# arguments on the CLI as the lookup data is necessarily very large
$: curl -H 'Moku-Client-Key: <key>' \
    -H 'Content-Type: application/json' \
    --data '{}' \
    http://<ip>/api/spectrumanalyzer/disable_xcorr
```

</code-block>
</code-group>
