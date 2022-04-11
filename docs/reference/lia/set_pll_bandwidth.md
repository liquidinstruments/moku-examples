---
additional_doc: The PLL in the LIA instrument is driven by the Moku's Input 2 and can optionally be used as a demodulation source. See `set_demodulation`.
description: Sets the tracking bandwidth of the PLL.
method: post
name: set_pll_bandwidth
parameters:
- default: null
  description: PLL Bandwidth
  name: bandwidth
  param_range: 10kHz, 2k5Hz, 600Hz, 150Hz, 40Hz, 10Hz
  type: string
  unit: null
- default: true
  description: Disable all implicit conversions and coercions.
  name: strict
  param_range: null
  type: boolean
  unit: null
summary: set_pll_bandwidth
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