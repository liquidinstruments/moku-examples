---
additional_doc: null
description: Configure the interpolation mode
method: post
name: set_interpolation
parameters:
    - default: SinX
      description: Set interpolation mode
      name: interpolation
      param_range: Linear, SinX, Gaussian
      type: string
      unit: null
    - default: true
      description: Disable all implicit conversions and coercions.
      name: strict
      param_range: null
      type: boolean
      unit: null
summary: set_interpolation
---

<headers/>
<parameters/>

<code-group>
<code-block title="Python">

```python
from moku.instruments import DigitalFilterBox
# Configure the instrument
i = DigitalFilterBox('192.168.###.###')

# Set linear interpolation
i.set_interpolation('Linear')
```

</code-block>

<code-block title="MATLAB">

```matlab
%% Configure the instrument
m = MokuDigitalFilterBox('192.168.###.###', true);

% Set linear interpolation
m.set_interpolation('Linear')

```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"interpolation": "Linear"}'\
        http://<ip>/api/digitalfilterbox/set_interpolation
```

</code-block>

</code-group>

### Sample response

```json
{
    "interpolation": "Linear"
}
```
