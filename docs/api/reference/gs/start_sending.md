---
additional_doc: null
description: Start transmitting packets for a duration, optionally with a trigger
method: post
name: start_sending
parameters:
    - default: null
      description: Duration to send (seconds)
      name: duration
      param_range: null
      type: double
      unit: Sec
    - default: 0
      description: Delay the start by (seconds)
      name: delay
      param_range: null
      type: integer
      unit: Sec
    - default: None
      description: Trigger source for starting the send
      name: trigger_source
      param_range: Input1, Input2, Input3, Input4, InputA, InputB, External
      type: string
      unit: null
    - default: None
      description: Trigger level in volts
      name: trigger_level
      param_range: -5V, 5V
      type: number
      unit: null
    - default: true
      description: Disable all implicit conversions and coercions.
      name: strict
      param_range: null
      type: boolean
      unit: null
summary: start_sending
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
gs.start_sending(duration=10, delay=1, strict=True)
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuMultiInstrument('192.168.###.###', 3)
gs = m.set_instrument(1, @MokuGigabitStreamer)
gs.start_sending(10, 1, 'strict', true)
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
       -H 'Content-Type: application/json'\
       --data '{"duration": 10, "delay": 1, "strict": true}'\
       http://<ip>/<slot>/api/gs/start_sending
```

</code-block>

</code-group>

### Sample response

```json
{
    "delay": "1",
    "duration": "10",
    "start_condition": "Delayed",
}
```
