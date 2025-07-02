---
additional_doc:
    When using either of the clients, user can access this function directly from
    instrument reference.
description: Configure the blended reference clock on the Moku
method: post
name: set_blended_clock
parameters:
    - default: null
      description: Enable external reference
      name: ext_ref_enable
      param_range: null
      type: boolean
      unit: null
    - default: null
      description: External reference frequency
      name: ext_ref_frequency
      param_range: 10MHz, 100MHz
      type: string
      unit: null
    - default: null
      description: Enable synchronization reference
      name: sync_ref_enable
      param_range: null
      type: boolean
      unit: null
    - default: null
      description: Source of synchronization reference
      name: sync_ref_source
      param_range: GNSS, Ext
      type: string
      unit: null
    - default: true
      description: Boolean flag representing the desired state of the clock
      name: enable
      param_range: null
      type: boolean
      unit: null
summary: set_blended_clock
available_on: 'Moku:Delta'
---

<headers/>

<parameters/>

### Examples

<code-group>
<code-block title="Python">

```python
i = Oscilloscope('192.168.###.###', force_connect=False)
# Here you can access the set_blended_clock function
i.set_blended_clock(ext_ref_enable=True, ext_ref_frequency='100MHz')
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuOscilloscope('192.168.###.###', false);

% Here you can access the set_blended_clock function
m.set_blended_clock('ext_ref_enable',True, 'ext_ref_frequency',"100MHz")

```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"ext_ref_enable": true, "ext_ref_frequency": "100MHz"}'\
        http://<ip>/api/moku/set_blended_clock
```

</code-block>

</code-group>
