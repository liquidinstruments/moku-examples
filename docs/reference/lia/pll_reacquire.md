---
additional_doc: null
description: Triggers a reacqusition cycle on the embedded PLL
method: post
name: pll_reacquire
parameters: []
summary: pll_reacquire
group: Input PLL
---

<headers/>
<parameters/>


<code-group>
<code-block title="Python">
```python
from moku.instruments import LockInAmp
i = LockInAmp('192.168.###.###')
i.pll_reacquire()
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuLockInAmp('192.168.###.###');
m.pll_reacquire()
```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{}'\
        http://<ip>/api/lockinamp/pll_reacquire
```
</code-block>
</code-group>

