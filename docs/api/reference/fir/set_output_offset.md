---
additional_doc: null
description: Set output signal offset
method: post
name: set_output_offset
parameters:
    - default: null
      description: Target channel
      name: channel
      param_range:
          mokugo: 1, 2
          mokulab: 1, 2
          mokupro: 1, 2, 3, 4
          mokudelta: 1, 2, 3, 4, 5, 6, 7, 8
      type: integer
      unit: null
    - default: null
      description: Output DC offset
      name: offset
      param_range:
          mokugo: -2.5 to 2.5
          mokulab: -1 to 1
          mokupro: -1 to 1
      type: number
      unit: V
    - default: true
      description: Disable all implicit conversions and coercions.
      name: strict
      param_range: null
      type: boolean
      unit: null
summary: set_output_offset
---

<headers/>
<parameters/>

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import FIRFilterBox
i = FIRFilterBox('192.168.###.###')
# Configure instrument to desired state
# Set output offset to 5VDC
i.set_output_offset(1, offset=5)
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuFIRFilterBox('192.168.###.###');
% Configure instrument to desired state
% Set output offset to 1VDC
m.set_output_offset(1, 'offset', 1);
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"channel": 1, "offset": 1}'\
        http://<ip>/api/pid/set_output_offset
```

</code-block>

</code-group>

### Sample response

```json
{ "offset": 1.0 }
```
