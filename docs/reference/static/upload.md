---
additional_doc: null
description: Upload file to the Moku file system
method: post
name: upload_file
parameters:
- default: null
  description: Target directory to access
  name: target
  param_range: ssd, persist, bitstream, media
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


User can upload files to files to **bitstream**, **persist**, **media** or **ssd** directories. If using the
REST API directly, these directories form the group name in the URL and the filename
follows the upload command; e.g. `/api/persist/upload/<filename>`. No Client Key is
required (ownership doesn't need to be taken).

When using either of the clients, user can access this function directly from
instrument reference.

#### bitstream Directory
Used for instrument logic. This is the destination to which one must upload Moku Cloud Compile
bitstreams before they can be deployed.

#### persist, ssd Directories
Data Logger output directory and general storage. It's rare that the user will want to load
files in to these locations, but the function is supported if required.

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

<code-block title="cURL">
```bash
$: curl --data @localfile.txt\
        http://<ip>/api/persist/upload/remotefile.txt
```
</code-block>


</code-group>


