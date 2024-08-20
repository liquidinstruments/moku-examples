---
additional_doc: This method is an overload of `set_source` which conveniently allows to configure multiple input channels through a single method call
description: Configures the signal source for multiple channels
method: post
name: set_sources
parameters:
    - default: null
      description: List of channel data sources
      name: sources
      param_range: null
      type: array
      unit: null
    - default: true
      description: Disable all implicit conversions and coercions.
      name: strict
      param_range: null
      type: boolean
      unit: null
summary: set_sources
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

# Set the data sources
i.set_sources([{"channel": 1, "source": "Output1"},
               {"channel": 2, "source": "Output2"}])
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuOscilloscope('192.168.###.###', true);
%% Configure the instrument
m.generate_waveform(1, 'Sine', 'amplitude',0.5, 'frequency', 10e3);
m.generate_waveform(2, 'Square', 'amplitude',1, 'frequency',20e3, 'duty', 50);

% Define data source 2
m1.channel = 1;
m1.source = 'Input1';

% Define data source 2
m2.channel = 2;
m2.source = 'Input2';

% Set data sources
m.set_sources([m1,m2]);

```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"sources": [{"channel": 1, "source": "Output1"},
                {"channel": 2, "source": "Output2"}]}'\
        http://<ip>/api/oscilloscope/set_sources
```

</code-block>

</code-group>

### Sample response

```json
{
    "sources": ["Output1", "Output2"]
}
```
