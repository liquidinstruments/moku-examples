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
      name: freq_ref_enable
      param_range: null
      type: boolean
      unit: null
    - default: false
      description: External reference frequency
      name: freq_ref_frequency
      param_range: 10MHz, 100MHz
      type: string
      unit: null
    - default: null
      description: Enable synchronization reference
      name: sync_ref_enable
      param_range: null
      type: boolean
      unit: null
    - default: false
      description: Source of synchronization reference
      name: sync_ref_source
      param_range: GNSS, Ext
      type: string
      unit: null
    - default: true
      description: Disable all implicit conversions and coercions.
      name: strict
      param_range: null
      type: boolean
      unit: null
summary: set_blended_clock
available_on: 'Moku:Delta'
---

<headers/>

::: warning Caution
Changes to the external frequency reference will not take effect until after your Moku:Delta is restarted.
:::

<parameters/>

### Examples

<code-group>
<code-block title="Python">

```python
i = Oscilloscope('192.168.###.###', force_connect=False)
# Here you can access the set_blended_clock function
i.set_blended_clock(freq_ref_enable=True, freq_ref_frequency='100MHz',
                    sync_ref_enable=False, sync_ref_source='GNSS')
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuOscilloscope('192.168.###.###', false);

% Here you can access the set_blended_clock function
m.set_blended_clock('freq_ref_enable',true, 'freq_ref_frequency','100MHz', ...
                    'sync_ref_enable',false, 'sync_ref_source','GNSS')

```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"freq_ref_enable": true, "freq_ref_frequency": "100MHz", "sync_ref_enable": false, "sync_ref_source": "GNSS"}'\
        http://<ip>/api/moku/set_blended_clock
```

</code-block>

</code-group>
