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
# Configure the demodulation path to follow the external PLL
i.set_demodulation(mode="ExternalPLL")
# Restart the PLL lock on the external reference signal
i.pll_reacquire()
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuLaserLockBox('192.168.###.###', force_connect=true);
% Configure the demodulation path to use the external PLL
m.set_demodulation('mode', 'ExternalPLL');
% Restart the PLL lock on the external reference signal
m.pll_reacquire()
```

</code-block>

<code-block title="cURL">

```bash
# Configure the demodulation path to follow the external PLL
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"mode": "ExternalPLL"}'\
        http://<ip>/api/laserlockbox/set_demodulation

# Restart the PLL lock on the external reference signal
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{}'\
        http://<ip>/api/laserlockbox/pll_reacquire
```

</code-block>
</code-group>
