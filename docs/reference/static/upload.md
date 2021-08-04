---
additional_doc: null
description: Upload file to the Moku file system
method: post
name: upload_file
parameters:
- default: null
  description: Target directory to access
  name: target
  param_range: bitstreams, persist
  type: string
  unit: null
- default: null
  description: Name of the file to upload
  name: file_name
  param_range: 
  type: string
  unit: null
- default: null
  description: Data to upload
  name: data
  param_range: 
  type: bytes
  unit: null
summary: upload_file
---


<headers/>


User can upload files to files to **persist** directory.

When using either of the clients, user can access this function directly from
instrument reference.

<parameters/>

Usage in clients,

<code-group>
<code-block title="Python">
```python
from moku.instruments import ArbitraryWaveformGenerator
i = ArbitraryWaveformGenerator('192.168.###.###', force_connect=False)
# Here you can access the upload_file function
i.upload_file(target='persist', file_name='sample.txt', data="Hello world!")
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuOscilloscope('192.168.###.###', false);
% Here you can access the upload_file function
m.upload_file('persist', 'sample.txt', 'Hello world!');
```
</code-block>
</code-group>


