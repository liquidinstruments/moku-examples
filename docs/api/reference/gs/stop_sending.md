---
additional_doc: null
description: Stop any ongoing packet transmission
method: post
name: stop_sending
parameters: []
summary: stop_sending
available_on: 'Moku:Delta'
---

<headers/>

When the Gigabit Streamer stops sending, either manually or at the end of a transmission, it will hold a DC value at the last known value. Disable all outputs to transmit no signal.

<parameters/>

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import MultiInstrument, GigabitStreamer
m = MultiInstrument('192.168.###.###', platform_id=3)
gs = m.set_instrument(1, GigabitStreamer)
gs.stop_sending()
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuMultiInstrument('192.168.###.###', 3)
gs = m.set_instrument(1, @MokuGigabitStreamer)
gs.stop_sending()
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
       -H 'Content-Type: application/json'\
       http://<ip>/<slot>/api/gs/stop_sending
```

</code-block>

</code-group>

### Sample response

```json
{
    []
}
```
