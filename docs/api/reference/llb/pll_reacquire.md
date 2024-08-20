---
additional_doc: null
description: Triggers a reacquisition cycle on the embedded PLL
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
from moku.instruments import LaserLockBox
i = LaserLockBox('192.168.###.###')
i.pll_reacquire()
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuLaserLockBox('192.168.###.###');
m.pll_reacquire()
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{}'\
        http://<ip>/api/laserlockbox/pll_reacquire
```

</code-block>
</code-group>
