---
additional_doc: null
description: Delete file from the Moku file system
method: delete
name: delete_file
parameters:
- default: null
  description: Target directory to access
  name: target
  param_range: bitstreams, persist
  type: string
  unit: null
- default: null
  description: Name of the file to delete
  name: file_name
  param_range: 
  type: string
  unit: null
---


<headers/>


User can delete files from **bitstreams** and **persist** directories.

When using either of the clients, user can access this function directly from
instrument reference.

<parameters/>

Usage in clients,

<code-group>
<code-block title="Python">
```python
from moku.instruments import ArbitraryWaveformGenerator
i = ArbitraryWaveformGenerator('192.168.###.###', force_connect=False)
# Here you can access the delete_file function
i.delete_file(target='persist', file_name='sample.txt')
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuOscilloscope('192.168.###.###', false);
% Here you can access the delete_file function
m.delete_file('persist', 'sample.txt');
```
</code-block>
</code-group>


