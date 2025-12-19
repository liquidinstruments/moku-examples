---
additional_doc: The Phase-Locked Loop (PLL) in Laser Lock Box is used when demodulation is set to External (PLL) mode. See set_demodulation for more information.
description: Sets the frequency acquisition/configuration and tracking bandwidth of the PLL.
method: post
name: set_pll
parameters:
    - default: true
      description: Auto acquire PLL frequency
      name: auto_acquire
      param_range: null
      type: boolean
      unit: null
    - default: undefined
      description: PLL frequency
      name: frequency
      param_range: null
      type: number
      unit: Hz
      warning: Setting frequency has no impact when auto_acquire is true
    - default: 1
      description: Frequency multiplier
      name: frequency_multiplier
      param_range: null
      type: number
      unit: null
    - default: 1kHz
      description: PLL Bandwidth
      name: bandwidth
      param_range: 1Hz, 10Hz, 100Hz, 1kHz, 10kHz, 100kHz, 1MHz
      type: string
      unit: null
    - default: true
      description: Disable all implicit conversions and coercions.
      name: strict
      param_range: null
      type: boolean
      unit: null
summary: set_pll
group: Input PLL
---

<headers/>
<parameters/>

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import LaserLockBox
i = LaserLockBox('192.168.###.###', force_connect=True)
# Disable auto-acquisition, set a 1 MHz target with 10 kHz tracking bandwidth
i.set_pll(auto_acquire=False, frequency=1e6, bandwidth="10kHz")
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuLaserLockBox('192.168.###.###', force_connect=true);
% Disable auto-acquisition, set a 1 MHz target with 10 kHz tracking bandwidth
m.set_pll('auto_acquire', false, 'frequency', 1e6, 'bandwidth', '10kHz');
```

</code-block>

<code-block title="cURL">

```bash
# Disable auto-acquisition, set a 1 MHz target with 10 kHz tracking bandwidth
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"auto_acquire": false, "frequency": 1000000, "bandwidth": "10kHz"}'\
        http://<ip>/api/laserlockbox/set_pll_bandwidth
```

</code-block>

</code-group>

### Sample response

```json
{
    "auto_acquire": false,
    "bandwidth": "10kHz",
    "frequency": 1000000.0,
    "frequency_multiplier": 1.0
}
```
