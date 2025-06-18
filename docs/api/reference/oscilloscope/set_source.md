---
additional_doc: null
description: Configures the signal source of each channel
method: post
name: set_source
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
      description: Set channel data source
      name: source
      param_range:
          mokugo: None, Input1, Input2, Output1, Output2
          mokulab: None, Input1, Input2, Output1, Output2
          mokupro: None, Input1, Input2, Input3, Input4, Output1, Output2, Output3, Output4
      type: string
      unit: null
    - default: true
      description: Disable all implicit conversions and coercions.
      name: strict
      param_range: null
      type: boolean
      unit: null
summary: set_source
---

<headers/>
<parameters/>

<code-group>
<code-block title="Python">

```python
from moku.instruments import Oscilloscope
# Configure the instrument
i = Oscilloscope('192.168.###.###')
i.generate_waveform(1, 'Sine', amplitude=0.5, frequency=10e3)
i.generate_waveform(2, 'Square', amplitude=1, frequency=20e3)

# Set the data source of Channel 1 to be Input 1
i.set_source(1,'Input1')
# Set the data source of Channel 2 to the generated output sinewave from Output 1
i.set_source(2,'Input2')
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuOscilloscope('192.168.###.###', true);
%% Configure the instrument
m.generate_waveform(1, 'Sine', 'amplitude',0.5, 'frequency', 10e3);
m.generate_waveform(2, 'Square', 'amplitude',1, 'frequency',20e3, 'duty', 50);

% Set the data source of Channel 1 to be Input 1
m.set_source(1,'Input1');
% Set the data source of Channel 2 to the generated output sinewave from
% Output 1
m.set_source(2,'Input2');

```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"channel": 2, "source": "Input2"}'\
        http://<ip>/api/oscilloscope/set_source
```

</code-block>

</code-group>

### Sample response

```json
{
    "source": "Input1"
}
```
