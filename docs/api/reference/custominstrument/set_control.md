---
additional_doc: null
description: Set a control register value
method: post
name: set_control
parameters:
    - default: null
      description: Target control register number
      name: idx
      param_range: 0 to 15
      type: integer
      unit: null
    - default: null
      description: Target control register value
      name: value
      type: integer
      unit: null
    - default: true
      description: Disable all implicit conversions and coercions.
      name: strict
      param_range: null
      type: boolean
      unit: null
summary: set_control
---

<headers/>
<parameters/>

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import MultiInstrument, CustomInstrument
m = MultiInstrument('192.168.###.###', platform_id=2)
bitstream = "path/to/project/adder/bitstreams.tar.gz"
cc = m.set_instrument(1, CustomInstrument, bitstream=bitstream)
# set connections
cc.set_control(2, 156)
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuMultiInstrument('192.168.###.###', 'platform_id', 2);
bitstream = 'path/to/project/adder/bitstreams.tar'
cc = m.set_instrument(1, @MokuCustomInstrument, bitstream);
% set connections
cc.set_control(2, 156);
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"idx": 2, "value": 156}'\
        http://<ip>/api/slot1/cloudcompile/set_control
```

</code-block>

</code-group>
