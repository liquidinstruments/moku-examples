---
additional_doc: It is strongly recommended that you call this at the completion of your script, including from
                error paths. Failure to do so means that device ownership is retained by the (finished) script
                and future users will need to forcefully take ownership before they can proceed.
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

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{}'\
        http://<ip>/api/moku/relinquish_ownership
```
</code-block>

</code-group>
