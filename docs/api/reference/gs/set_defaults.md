---
additional_doc: null
description: Reset the instrument to its default configuration
method: post
name: set_defaults
parameters: []
summary: set_defaults
available_on: 'Moku:Delta'
---

<headers/>
<parameters/>

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import MultiInstrument, GigabitStreamer 
m = MultiInstrument('192.168.###.###', platform_id=3)
gs = m.set_instrument(1, GigabitStreamer)
gs.set_defaults()
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuMultiInstrument('192.168.###.###', 3)
gs = m.set_instrument(1, @MokuGigabitStreamer)
gs.set_defaults()
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
       -H 'Content-Type: application/json'\
       http://<ip>/<slot>/api/gs/set_defaults
```

</code-block>

</code-group>
