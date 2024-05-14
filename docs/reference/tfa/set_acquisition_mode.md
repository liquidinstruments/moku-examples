---
additional_doc: The acquisition mode of the Time & Frequency Analyzer is the acquisition of intervals, which therefore dictates how the instrument collects interval data
description: Configures acquisition mode to determine how statistics, histograms and output signals are computed.
method: post
name: set_acquisition_mode
parameters:
- default: Continuous
  description: Acquisition mode
  name: mode
  param_range: Continuous ,Windowed, Gated
  type: string
  unit: 
- default: 
  description: Acquisition channel
  name: gate_source
  param_range:
   mokugo: ChannelA, ChannelB, Input1, Input2, Output1, Output2
   mokulab: ChannelA, ChannelB, Input1, Input2, Output1, Output2, External
   mokupro: ChannelA, ChannelB, ChannelC, ChannelD, Input1, Input2, Input3, Input4, Output1, Output2, Output3, Output4, External
  type: string
  unit: 
- default: 
  description: Acquisition threshold
  name: gate_threshold
  param_range: 
  type: number
  unit: 
- default: 
  description: Window length
  name: window_length
  param_range: 
  type: number
  unit: 
- default: True
  description: Disable all implicit conversions and coercions.
  name: strict
  param_range: 
  type: boolean
  unit: 
summary: set_acquisition_mode
---


<headers/>

<parameters/>

### Windowed acquisition
Windowed acquisition counts events within each window period, and resets at the end of each window
period. Windowed proves a balance of averaging for noise reduction and the ability to respond when
the signal changes.
### Continuous acquisition
Events are being registered continuously during intervals.
### Gated acquisition
Similar to windowed acquisition, statistics, count output and histogram counts are reset at the end of
each gate period. In gated acquisition mode, the beginning of the acquisition period is triggered by the
gate input signal. The gate input source can be any of the Moku device input channels, or by external
trigger (if available).



### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import TimeFrequencyAnalyzer
i = TimeFrequencyAnalyzer('192.168.###.###')
# configure event detectors
# configure interval analyzers
# configure acquisition mode
i.set_acquisition_mode(mode="Continuous")
# retrieve data 
data = i.get_data()
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuTimeFrequencyAnalyzer('192.168.###.###')
% configure event detectors
% configure interval analyzers
% configure acquisition mode
m.set_acquisition_mode('Continuous')
% retrieve data 
m.get_data()
```

</code-block>

<code-block title="cURL">

```bash
# You should create a JSON file with the data content rather than passing
# arguments on the CLI as the lookup data is necessarily very large
$: cat request.json
{
    "mode": "Continuous"
}
$: curl -H 'Moku-Client-Key: <key>' \
    -H 'Content-Type: application/json' \
    --data @request.json \
    http:// <ip >/api/tfa/set_acquisition_mode
```

</code-block>

</code-group>
