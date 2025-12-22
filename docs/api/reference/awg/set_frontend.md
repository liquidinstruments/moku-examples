---
additional_doc: Setting the bandwidth is only available on Moku:Pro. Read about [how to select the bandwidth for your application](../README.md#bandwidth)
description: Configures the input impedance, coupling, and range for each channel
method: post
name: set_frontend
parameters:
    - default: null
      description: Target channel
      name: channel
      param_range:
          mokugo: 1, 2
          mokulab: 1, 2
          mokupro: 1, 2, 3, 4
          mokudelta: 1, 2, 3, 4, 5, 6, 7, 8
      type: integer
      unit: null
    - default: null
      description: Impedance
      name: impedance
      param_range:
          mokugo: 1MOhm
          mokulab: 50Ohm, 1MOhm
          mokupro: 50Ohm, 1MOhm
          mokudelta: 50Ohm, 1MOhm
      type: string
      unit: null
    - default: null
      description: Input Coupling
      name: coupling
      param_range: AC, DC
      type: string
      unit: null
    - default: null
      description: Input Range
      name: range
      param_range:
          mokugo: 10Vpp, 50Vpp
          mokulab: 1Vpp, 10Vpp
          mokupro: 400mVpp, 4Vpp, 40Vpp
          mokudelta: 100mVpp, 1Vpp, 10Vpp, 40Vpp
      type: string
      unit: null
    - default: 300MHz
      description: Input bandwidth
      name: bandwidth
      param_range:
          mokupro: 300MHz, 600MHz
      type: string
      unit: null
    - default: true
      description: Disable all implicit conversions and coercions.
      name: strict
      param_range: null
      type: boolean
      unit: null
summary: set_frontend
---

<headers/>
<parameters/>

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import ArbitraryWaveformGenerator
# Connect to your Moku
i = ArbitraryWaveformGenerator('192.168.###.###', force_connect=True)

# As input channels are only used when there is a modulation, we will use burst mode in this example
# Add burst modulation to waveform on channel 1, triggering on Input 1 at 0.1V level for 3 cycles
i.burst_modulate(1, "Input1", "NCycle", burst_cycles=3, trigger_level=0.1)
# Set Input 1 to 1 MOhm, DC coupled, 4 Vpp input range, and 300 MHz bandwidth
i.set_frontend(channel=1, impedance="1MOhm", coupling="DC", range="4Vpp", bandwidth="300MHz", strict=True)
```

</code-block>

<code-block title="MATLAB">

```matlab
% Connect to your Moku
m = MokuArbitraryWaveformGenerator('192.168.###.###', force_connect=true);

% As input channels are only used when there is a modulation, we will use burst mode in this example
% Add burst modulation to waveform on channel 1, triggering on Input 1 at 0.1V level for 3 cycles
m.burst_modulate(2, "Input1", "NCycle", 'burst_cycles', 3, 'trigger_level', 0.1);
% Set Input 1 to 1 MOhm, DC coupled, 4 Vpp input range, 300 MHz bandwidth
m.set_frontend(1, '1MOhm', 'DC', '4Vpp', 'bandwidth', '300MHz', 'strict', true);
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{'strict': True, 'channel': 1, 'impedance': '1MOhm', 'coupling': 'DC', 'range': '4Vpp', 'bandwidth': '300MHz'}'\
        http://<ip>/api/awg/set_frontend
```

</code-block>

</code-group>

### Sample response

```json
{
    "bandwidth": "300MHz",
    "coupling": "DC",
    "impedance": "1MOhm",
    "range": "4Vpp"
}
```
