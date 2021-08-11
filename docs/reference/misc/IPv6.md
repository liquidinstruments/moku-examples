---
additional_doc: 
description: 
method: post
name: IPv6 instructions
summary: How to connect to your Moku:Go via USB
---

<headers/>

You can connect to your Moku:Go via USB. The USB connection uses IPv6 and you can copy the 
IP address from our Moku App.

The syntax for connecting to your Moku is shown below:

<code-group>
<code-block title="Python">
```python{6,7}
from moku.instruments import LogicAnalyzer
i = LogicAnalyzer('[IPv6_address]', force_connect=False)

```
</code-block>

<code-block title="MATLAB">
```matlab{5,6}
m = MokuLogicAnalyzer('[IPv6_address]', true);
```
</code-block>
</code-group>

