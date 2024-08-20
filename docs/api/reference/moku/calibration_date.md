---
title: calibration_date
additional_doc: null
description: Get the last calibration date of the Moku.
method: get
name: calibration_date
parameters: []
summary: calibration_date
---

<headers/>

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import ArbitraryWaveformGenerator

i = ArbitraryWaveformGenerator('192.168.###.###', force_connect=False)

# Here you can access the name function
i.calibration_date()
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuOscilloscope('192.168.###.###', false);

% Here you can access the name function
m.calibration_date()

```

</code-block>

<code-block title="cURL">

```bash
$: curl http://<ip>/api/moku/calibration_date
```

</code-block>

</code-group>
