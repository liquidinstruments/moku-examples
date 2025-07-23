---
additional_doc: null
description: Check if the Moku is using the blended reference clock
method: get
name: get_blended_clock
parameters: []
summary: get_blended_clock
available_on: 'Moku:Delta'
---

<headers/>

<parameters/>

### Examples

<code-group>

<code-block title="Python">

```python
i = Oscilloscope('192.168.###.###', force_connect=False)
# Here you can access the get_blended_clock function
i.get_blended_clock()
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuOscilloscope('192.168.###.###', false);

% Here you can access the get_blended_clock function
m.get_blended_clock()

```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        http://<ip>/api/moku/get_blended_clock
```

</code-block>

</code-group>
