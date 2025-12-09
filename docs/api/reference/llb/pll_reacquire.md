---
additional_doc: null
description: Restarts the PLL on the external reference signal. Acquisition uses an FFT-based frequency estimate.
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
i = LaserLockBox('192.168.###.###', force_connect=True)
i.pll_reacquire()
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuLaserLockBox('192.168.###.###', force_connect=true);
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
