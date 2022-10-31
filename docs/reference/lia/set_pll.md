---
additional_doc: The PLL in the LIA instrument is driven by the Moku's Input 2 and can optionally be used as a demodulation source. See `set_demodulation`.
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
  warning: Setting frequency has no impact when auto_acquire is true
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
from moku.instruments import LockInAmp
i = LockInAmp('192.168.###.###')
i.set_pll_bandwidth("10kHz")
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuLockInAmp('192.168.###.###');
m.set_pll_bandwidth('bandwidth','10kHz');
```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"bandwidth": "10kHz"}'\
        http://<ip>/api/lockinamp/set_pll_bandwidth
```
</code-block>

</code-group>

### Sample response
```json
{
  "bandwidth": "10kHz"
}
```