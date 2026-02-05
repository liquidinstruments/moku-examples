---
additional_doc:
    There is a single PID block internally that may be connected to either channel.
    The PID parameters can be configured with `set_by_frequency`.
description: Enables and disables a PID controller on either the Main or Aux outputs
method: post
name: use_pid
parameters:
    - default: null
      description: Which channel, if any, to PID control
      name: channel
      param_range: Main, Aux, Off
      type: string
      unit: null
    - default: true
      description: Disable all implicit conversions and coercions.
      name: strict
      param_range: null
      type: boolean
      unit: null
summary: use_pid
group: PID Controller
---

<headers/>

When using `Internal` and `External PLL` [set_demodulation](./set_demodulation.md) modes,
the PID controller [use_pid](./use_pid.md) can always be attached to the `main`
output. However, the PID controller can be attached to either the `main` or `aux`
sources when set_outputs is not outputting a demodulation source (`Demod`) or an
auxiliary oscillator (`Aux`).

When using `External` and `None` [set_demodulation](./set_demodulation.md) modes, the PID
controller can only be attached to the `main` output using [use_pid](./use_pid.md). Further,
when using these demodulation modes, the `aux` output can only be set to `Aux` oscillator.


<parameters/>

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import LockInAmp
i = LockInAmp('192.168.###.###')
i.use_pid("Main")
i.set_by_frequency(prop_gain=-10)

```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuLockInAmp('192.168.###.###');
m.use_pid('channel',"Main");
m.set_by_frequency('prop_gain',10);
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"channel": "Main"}'\
        http://<ip>/api/lockinamp/use_pid
```

</code-block>

</code-group>

### Sample response

```json
{
    "channel": "Main"
}
```
