---
description: Powerdown the Moku.
method: post
name: shutdown
summary: shutdown
parameters: []
---

<headers/>

This powers off your Moku remotely. It cannot be turned back on remotely, use 'reboot' to powercycle your Moku. 

::: caution
This function is implimented in Python only at the moment.
:::

<parameters/>

Examples:

<code-group>
<code-block title="Python">

```python
from moku.instruments import ArbitraryWaveformGenerator

i = ArbitraryWaveformGenerator('192.168.###.###', force_connect=False)
# Shurdown the Moku
i.shutdown()
```

</code-block>

<code-block title="MATLAB">

```matlab
i = MokuOscilloscope('192.168.###.###', false);

% Shutdown the Moku
i.shutdown()

```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{}'\
        http://<ip>/api/moku/shutdown
```

</code-block>

</code-group>

