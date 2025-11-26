---
title: platform
additional_doc: It is required to configure the platform to use moku in Multi-Instrument Mode or switch back to single-instrument mode
description: Configure the platform ID
method: get
name: platform
path_parameters:
    - default: null
      description: ID of the platform to deploy
      name: id
      param_range:
          mokugo: 1, 2, 3
          mokulab: 1, 2, 3
          mokupro: 1, 4
          mokudelta: 1, 3, 8
      type: integer
      unit: null
summary: platform
---

<headers/>
<path-parameters/>

:::tip
When using either of the clients, `id` is passed as an argument(`platform_id`) to MultiInstrument class, refer to examples below
:::

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import MultiInstrument
m = MultiInstrument('192.168.###.###', platform_id=2, force_connect=True)
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuOscilloscope('192.168.###.###', false);

% Here you can access the name function
m.name()

```

</code-block>

<code-block title="cURL">

```bash
$: curl http://<ip>/api/moku/platform/4
```

</code-block>

</code-group>
