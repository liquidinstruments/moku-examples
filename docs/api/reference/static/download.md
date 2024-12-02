---
additional_doc: null
description: Download file from the Moku file system to local machine
method: get
name: download_file
parameters:
    - default: null
      description: Target directory to access
      name: target
      param_range: 
          mokugo: persist, bitstreams, logs, tmp
          mokulab: media, bitstreams, logs, tmp
          mokupro: ssd, bitstreams, logs, tmp, persist
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

User can download files from **bitstreams**, **logs**, **ssd**, **media** and **persist** directories. If using the REST API directly, these directories form the group name in the URL and the filename follows the download command; e.g. `/api/persist/download/<filename>`. No Client Key is required (ownership doesn't need to be taken).

When using either of the clients, user can access this function directly from instrument reference.

Log files downloaded from `persist` (Moku:Go), `media` (Moku:Lab) or `ssd` (Moku:Pro) will be in _.li format, which can then be converted to _.csv, _.mat, or _.npy using LI File Converter.

If you experience issues converting log files with the MATLAB API, please use **download_file2** method with the same parameters.

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

<code-block title="cURL">

```bash
$: curl http://<ip>/api/persist/download/remotefile.txt -o localfile.txt
```

</code-block>

</code-group>
