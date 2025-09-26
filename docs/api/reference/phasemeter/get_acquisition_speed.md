---
additional_doc: null
description: Get current acquisition speed
method: get
name: get_acquisition_speed
parameters: []
summary: get_acquisition_speed
---

<headers/>
There are 6 different acquisition speeds available in the Phasemeter instrument, the response from the Moku matches the following list:

-   -19 = 37 Hz
-   -17 = 150 Hz
-   -15 = 596 Hz
-   -13 = 2.4 kHz
-   -10 = 19.1 kHz
-   -7 = 152 kHz

<parameters/>

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import Phasemeter

i = Phasemeter('192.168.###.###', force_connect=True)

# Request the acquisition speed of the Phasemeter
speed = i.get_acquisition_speed()
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuPhasemeter('192.168.###.###', force_connect=true);

% Request the acquisition speed of the Phasemeter
speed = m.get_acquisition_speed()

```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        http://<ip>/api/phasemeter/get_acquisition_speed
```

</code-block>

</code-group>
