---
additional_doc: null
description: Configures the desired output channel
method: post
name: set_output
parameters:
    - default: null
      description: Target output channel to configure, where channel 1 is the Fast controller path and channel 2 is the Slow controller path
      name: channel
      param_range: 1, 2
      type: integer
      unit: null
    - default: null
      description: Engage or disengage the control signal switch.
      name: signal
      param_range: null
      type: boolean
      unit: null
    - default: null
      description: Enable or disable the output port
      name: output
      param_range: null
      type: boolean
      unit: null
    - default: 0dB
      description: Output gain range
      name: gain
      param_range:
        mokugo: 0dB
        mokulab: 0dB
        mokupro: 0dB, 14dB
        mokudelta: 0dB, 20dB
      type: string
      unit: null
    - default: true
      description: Disable all implicit conversions and coercions.
      name: strict
      param_range: null
      type: boolean
      unit: null
summary: set_output
---

<headers/>
<parameters/>

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import LaserLockBox
i = LaserLockBox('192.168.###.###', force_connect=True)
# Configure the fast controller path with 0 dB gain:
# connect the fast controller switch, enable the output channel, and keep unity gain
i.set_output(1, signal=True, output=True, gain_range="0dB")

# Configure the slow controller path with 0 dB gain:
# connect the slow controller switch, enable the output channel, and keep unity gain
i.set_output(2, signal=True, output=True, gain_range="0dB")
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuLaserLockBox('192.168.###.###', force_connect=true);
% Configure the fast controller path with 0 dB gain:
% connect the fast controller switch, enable the output channel, and keep unity gain
m.set_output(1, 'signal', true, 'output', true, 'gain_range', '0dB');

% Configure the slow controller path with 0 dB gain:
% connect the slow controller switch, enable the output channel, and keep unity gain
m.set_output(2, 'signal', true, 'output', true, 'gain_range', '0dB');
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"output": true, "signal": true, "gain_range": "0dB"}'\
        http://<ip>/api/laserlockbox/set_output
```

</code-block>

</code-group>

### Sample response

```json
{
    "gain_range": "0dB",
    "output": true,
    "signal": true
}
```
