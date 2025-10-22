---
additional_doc: null
description: Delete file from the Moku file system
method: delete
name: delete_file
parameters:
    - default: null
      description: Target directory to access
      name: target
      param_range:
          mokugo: persist, bitstreams, logs, tmp
          mokulab: media, bitstreams, logs, tmp
          mokupro: ssd, bitstreams, logs, tmp, persist
          mokudelta: ssd, bitstreams, logs, tmp, persist
      type: string
      unit: null
    - default: null
      description: Name of the file to delete
      name: file_name
      param_range:
      type: string
      unit: null
summary: delete_file
---

<headers/>

User can delete files from **bitstreams**, **ssd**, **logs**, **persist** and **media** directories. If using the REST API directly, these directories form the group name in the URL and the filename
follows the delete command; e.g. `/api/persist/delete/<filename>`. No Client Key is required (ownership doesn't need to be taken).

When using either of the clients, user can access this function directly from instrument reference.

<parameters/>

:::tip Python Support
In Python the command is delete(), MATLAB and cURL use the command delete_file()
:::

Usage in clients,

<code-group>
<code-block title="Python">

```python
from moku.instruments import ArbitraryWaveformGenerator
i = ArbitraryWaveformGenerator('192.168.###.###', force_connect=False)
# Here you can access the delete_file function
i.delete(target='persist', file_name='sample.txt')
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuOscilloscope('192.168.###.###', false);
% Here you can access the delete_file function
m.delete_file('persist', 'sample.txt');
```

</code-block>

<code-block title="cURL">

```bash
$: curl -X DELETE
        http://<ip>/api/persist/delete_file/remotefile.txt
```

</code-block>

</code-group>
