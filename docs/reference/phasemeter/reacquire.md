---
additional_doc: null
description: Reacquire phasemeter signals.
method: post
name: reacquire
parameters: null
summary: reacquire
available_on: "mokupro"
---

<headers/>
<parameters/>

### Examples

<code-group>
<code-block title="Python">
```python
from moku.instruments import Phasemeter

i = Phasemeter('192.168.###.###')
i.reacquire()
```
</code-block>

<code-block title="MATLAB">
```matlab
i = MokuPhasemeter('192.168.###.###');
i.reacquire();
```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"disable": true}'\
        http://<ip>/api/phasemeter/reacquire
```
</code-block>

</code-group>

