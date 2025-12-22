---
additional_doc: null
description: Reacquire phasemeter signals.
method: post
name: reacquire
parameters: []
summary: reacquire
---

<headers/>
<parameters/>

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import Phasemeter

i = Phasemeter('192.168.###.###', force_connect=True)
i.reacquire()
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuPhasemeter('192.168.###.###', force_connect=true);
m.reacquire();
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
