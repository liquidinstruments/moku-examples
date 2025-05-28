---
additional_doc: null
description: Returns a short summary of current instrument state
method: get
name: summary
parameters: []
summary: summary
---

<headers/>
<parameters/>

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import MultiInstrument, CloudCompile
m = MultiInstrument('192.168.###.###', platform_id=2)
cc = m.set_instrument(1, CloudCompile)
# set connections
print(cc.summary())
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuMultiInstrument('192.168.###.###', 'platform_id', 2);
cc = m.set_instrument(1, MokuCloudCompile);
% set connections
disp(cc.summary())
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        http://<ip>/api/slot1/cloudcompile/summary
```

</code-block>

</code-group>

### Sample response

```plaintext
Moku:Go Cloud Compile
Register 0: 0
Register 1: 0
Register 2: 0
Register 3: 0
Register 4: 0
Register 5: 0
Register 6: 0
Register 7: 0
Register 8: 0
Register 9: 0
Register 10: 0
Register 11: 0
Register 12: 0
Register 13: 0
Register 14: 0
Register 15: 0
```
