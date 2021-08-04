---
additional_doc: null
description: List files from the Moku file system
method: get
name: list_files
parameters:
- default: null
  description: Target directory to access
  name: target
  param_range: bitstreams, logs, persist
  type: string
  unit: null
summary: list_files
---


<headers/>


User can list files from **bitstreams**, **logs** and **persist** directories.

When using either of the clients, user can access this function directly from
instrument reference.

<parameters/>

Usage in clients,

<code-group>
<code-block title="Python">
```python
from moku.instruments import ArbitraryWaveformGenerator

i = ArbitraryWaveformGenerator('192.168.###.###', force_connect=False)

# Here you can access the list function
i.list(target='persist')
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuOscilloscope('192.168.###.###', false);

% Here you can access the list_files function
m.list_files('target', 'persist')
```
</code-block>
</code-group>


