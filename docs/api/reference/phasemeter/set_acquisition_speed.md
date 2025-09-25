---
additional_doc: null
description: Sets the acquisition speed of the instrument
method: post
name: set_acquisition_speed
parameters:
    - default: null
      description: Target samples per second
      name: speed
      param_range:
          mokugo: 30Hz, 119Hz, 477Hz, 1.9kHz, 15.2kHz, 122kHz
          mokulab: 30Hz, 119Hz, 477Hz, 1.9kHz, 15.2kHz
          mokupro: 37Hz, 150Hz, 596Hz, 2.4kHz, 19.1kHz, 152kHz
          mokudelta: 37Hz, 150Hz, 596Hz, 2.4kHz, 19.1kHz, 152kHz
      type: string
      unit: null
    - default: true
      description: Disable all implicit conversions and coercions.
      name: strict
      param_range: null
      type: boolean
      unit: null
summary: set_acquisition_speed
---

<headers/>
<parameters/>

### Examples

<code-group>
<code-block title="Python">

```python{6}
from moku.instruments import Phasemeter
i = Phasemeter('192.168.###.###', force_connect=True)
# Set required acquisition speed
i.set_acquisition_speed(speed='596Hz')
```

</code-block>

<code-block title="MATLAB">

```matlab{8}
i = MokuPhasemeter('192.168.###.###', force_connect=true);
# Set required acquisition speed
i.set_acquisition_speed('19.1kHz');
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"speed": "19.1kHz"}'\
        http://<ip>/api/phasemeter/set_acquisition_speed
```

</code-block>

</code-group>
