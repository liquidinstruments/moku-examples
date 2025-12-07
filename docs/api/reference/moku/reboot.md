---
description: Reboot the Moku.
method: post
name: reboot
summary: reboot
parameters: []
---

<headers/>

This reboots, or power cycles, your Moku remotely. 

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
i.reboot()
```

</code-block>
</code-group>
