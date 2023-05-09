---
additional_doc: null
description: Configures the burst modulation mode of a channel
method: post
name: burst_modulate
parameters:
- default: null
  description: Target channel
  name: channel
  param_range:
   mokugo: 1, 2
   mokulab: 1, 2
   mokupro: 1, 2, 3, 4
  type: integer
  unit: null
- default: null
  description: Trigger source
  name: trigger_source
  param_range: 
    mokugo: Input1, Input2
    mokulab: Input1, Input2, External
    mokupro: Input1, Input2, Input3, Input4, External  
  type: string
  unit: null
- default: null
  description: Burst mode
  name: trigger_mode
  param_range: Start, NCycle
  type: string
  unit: null
- default: 1
  description: Number of cycles to generate when triggered
  name: burst_cycles
  param_range: 1 to 1e6
  type: number
  unit: null
- default: 0
  description: Trigger level
  name: trigger_level
  param_range: 
   mokugo: -5 to 5
   mokulab: -5 to 5
   mokupro: -20 to 20
  type: number
  unit: V
- default: 10Vpp for Go and 400mVpp for Pro
  description: Input Range
  name: input_range
  param_range: 
   mokugo: 10Vpp, 50Vpp
   mokulab: 10Vpp, 1Vpp
   mokupro: 400mVpp, 4Vpp, 40Vpp 
  type: string
  unit: Vpp
- default: true
  description: Disable all implicit conversions and coercions.
  name: strict
  param_range: null
  type: boolean
  unit: null
summary: burst_modulate
---

<headers/>
<parameters/>

### Examples

<code-group>
<code-block title="Python">
```python
from moku.instruments import ArbitraryWaveformGenerator
i = ArbitraryWaveformGenerator('192.168.###.###')
# Configure the output waveform in each channel
# Set Channel 2 to burst mode
# Burst mode triggering from Input 1 at 0.1 V
# 3 cycles of the waveform will be generated every time it is triggered
i.burst_modulate(2, "Input1", "NCycle", burst_cycles=3, trigger_level=0.1)
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuArbitraryWaveformGenerator('192.168.###.###');
% Configure the output waveform in each channel

%% Set Channel 2 to burst mode
% Burst mode triggering from Input 1 at 0.1 V
% 3 cycles of the waveform will be generated every time it is triggered
m.burst_modulate(2, "Input1", "NCycle",'burst_cycles',3,'trigger_level',0.1);
```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"channel":2, "trigger_source": "Input1", "trigger_mode": "NCycle", "burst_cycles": 3, "trigger_level": 0.1}'\
        http://<ip>/api/awg/burst_modulate
```
</code-block>


</code-group>

### Sample Response
```json
{
  "burst_cycles":1,
  "input_range":"10Vpp",
  "trigger_level":0.0,
  "trigger_mode":"Start",
  "trigger_source":"Input1"
}
```

