---
description: Powerdown the Moku.
method: post
name: shutdown
summary: shutdown
parameters: []
---

<headers/>

This powers off your Moku remotely. It cannot be turned back on remotely, use 'reboot' to powercycle your Moku. 

::: tip INFO
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

</code-group>

