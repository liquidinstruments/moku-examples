---
additional_doc: null
description: Relinquish the ownership of the Moku. 
method: post
name: relinquish_ownership
summary: relinquish_ownership
parameters: []
---

<headers/>

This closes the connection with your Moku and allows it to be connected by another client (desktop app, iPad, or API).

<parameters/>

Examples:

<code-group>
<code-block title="Python">
```python
from moku.instruments import ArbitraryWaveformGenerator

i = ArbitraryWaveformGenerator('192.168.###.###', force_connect=False)
# Close the current session with the Moku 
i.relinquish_ownership()
```
</code-block>

<code-block title="MATLAB">
```matlab
i = MokuOscilloscope('192.168.###.###', false);

% Close the current session with the Moku 
i.relinquish_ownership()
```
</code-block>
</code-group>
