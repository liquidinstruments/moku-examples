---
title: name
additional_doc: When using either of the clients, user can access this function directly from
                instrument reference.
description: Get name of the Moku
method: get
name: name
parameters: []
summary: name
---


<headers/>

Examples,

<code-group>
<code-block title="Python">
```python
from moku.instruments import ArbitraryWaveformGenerator

i = ArbitraryWaveformGenerator('192.168.###.###', force_connect=False)

# Here you can access the name function
i.name()
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuOscilloscope('192.168.###.###', false);

% Here you can access the name function
m.name()
```
</code-block>
</code-group>