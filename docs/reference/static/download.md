---
additional_doc: null
description: Download file from the Moku file system to local machine
method: get
name: download_file
parameters:
- default: null
  description: Target directory to access
  name: target
  param_range: bitstreams, logs, persist
  type: string
  unit: null
- default: null
  description: Name of the file to download
  name: file_name
  param_range: 
  type: string
  unit: null
- default: null
  description: Local path to download the file
  name: local_path
  param_range: 
  type: string
  unit: null
  warning: local_path should always be a absolute path with file name. For example, in Python instead of passing it as "/user/home" or "./home" pass it as "/user/home/sample.txt"
summary: download_file
---


<headers/>


User can download files from **bitstreams**, **logs** and **persist** directories.

When using either of the clients, user can access this function directly from
instrument reference.

<parameters/>

Usage in clients,

<code-group>
<code-block title="Python">
```python
from moku.instruments import ArbitraryWaveformGenerator
i = ArbitraryWaveformGenerator('192.168.###.###', force_connect=False)
# Here you can access the download_file function
i.download_file(target='persist', file_name='sample.txt', 
    local_path="path to local directory")
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuOscilloscope('192.168.###.###', false);
% Here you can access the download_file function
m.download_file('persist', 'sample.txt', 'path to local directory');
```
</code-block>
</code-group>


