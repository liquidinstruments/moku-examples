---
---
# Getting Started with Python
If at any point, your output doesn't match what's listed below, please refer to the [Troubleshooting](#troubleshooting) section

## Requirements
- Python 3.5+
- Knowledge of your Moku's IP Address (see below)

## Quickstart
If you're comfortable installing Python packages, then all you need to know is in this snippet. If not, or if you run in to any trouble, please use the full steps below.
```
$: pip install moku
$: moku download
$: python
>>> from moku.instruments import Oscilloscope
>>> osc = Oscilloscope('192.168.123.45')
>>> osc.get_data()
```
You should see an array of data captured from your newly-deployed Oscilloscope object. If not, please refer to the Troubleshooting below.

## Full Procedure
### 1. Check your Python Installation
At a command prompt (e.g. cmd.exe, Windows Terminal, MacOS Terminal) check your Python version. It should be greater than or equal to `3.5.0`. In particular, Python 2 (e.g. `2.7.0`) is not supported.

```
$: python --version
Python 3.9.0
```

### 2. Install the `moku` Library
At a command prompt, install the `moku` library using `pip`. You can easily check that the installation succeeded by running the simple Python command listed below. If there is *no output* from the Python command, then the installation has succeeded. If you see an error message, refer to the [Troubleshooting](#troubleshooting) section below.
```
$: pip install moku
$: python -c 'import moku'
$:
```

### 3. Download the data files
The Moku Scripting API for Python requires data files to be downloaded before any program can be run. These data files may be several hundred megabytes. Please ensure you have a suitable internet connection before you proceed, this step is only required to be run whenever you install or upgrade the library.

At a terminal or command prompt, issue the download command. This may take a while to complete, depending on your internet connection.
```
$: moku download
```

### 4. Find Your IP Address
For more detail and other options, see [Finding your IP Address](ip-address.html)

```
$: moku list
```

### 5. Run Your First Program
Use your IP address as found above in place of the one below.
```python
$: python
>>> from moku.instruments import Oscilloscope
>>> osc = Oscilloscope('192.168.123.45')
>>> osc.get_data()
```
You should see an array of data captured from your newly-deployed Oscilloscope object. If not, please refer to the Troubleshooting below.

## Next Steps
Check out our [Python Examples](/examples/python/) for more inspiration. Happy Coding!

## Troubleshooting

#### 'python' is not recognized as an internal or external command, operable program or batch file.
Python is not correctly installed on your Windows machine. Follow the instructions [here](https://docs.python.org/3/using/windows.html).

#### python: command not found
Python is not correctly installed on your Mac, or you're using a UNIX terminal on Windows that also cannot find Python. For Mac, follow the instructions [here](https://docs.python.org/3/using/mac.html).

#### The system cannot find the file C:\Users\user\AppData\Local\Microsoft\WindowsApps\python.exe.
Newer versions of Windows ship with Python installed by default. This error message indicates that the default installation has become corrupted somehow. Liquid Instruments recommends installing Python from scratch, according to the instructions [here](https://docs.python.org/3/using/windows.html).

#### ModuleNotFoundError: No module named 'pip'
You have tried to execute the `pip` command from within Python, rather that from a command prompt. Try using a terminal like Windows Terminal, Mac Terminal or `cmd.exe`.

#### Listed Version is less than 3.5
You must upgrade your Python to 3.5 or newer. See the installation instructions for [Windows](https://docs.python.org/3/using/windows.html) or [Mac](https://docs.python.org/3/using/mac.html)

#### ModuleNotFoundError: No module named 'moku'
Python cannot find the newly-installed `moku` library. This commonly occurs for one of two reasons:
- The `pip` command didn't complete successfully, OR
- A problem in your Python installation means that your `pip` and `python` executables have got out of sync with each other.

If you are sure that the `pip` command completed successfully, then you can resolve the problem option by executing `pip` directly from the Python installation, rather than separately.
```
$: python -m pip install moku
$: python -c 'import moku'
$:
```

#### Connection to (ip address) timed out, Max retries exceeded
The Moku library was unable to connect to the Moku device. Ensure you have the correct IP address using the steps above.

If the IP address is correct, check that the network is reachable from your computer. The easiest way to achieve this is to simply try and connect using the Moku Desktop Application.

#### USB Connection Issues
On Windows (only, not Mac or Linux), the Moku requires a driver to be installed in order to work over USB. This driver is installed automatically when you install the Moku Desktop Application. If you are working in an environment where you are not able to install the full Desktop Application, a standalone driver installer is available by contacting our [Support Engineers](mailto:support@liquidinstruments.com).

The Moku API uses IPv6 to connect to the Moku over USB. If your computer does not have IPv6 enabled (e.g. disabled by an Administrator) then the USB connection will not be operable. Please reach out to our [Support Engineers](mailto:support@liquidinstruments.com) for guidance and other options.

This extends to environments inside your computer where IPv6 may be limited, such as Microsoft Windows Subsystem for Linux v2 (WSL2). For information on possible workarounds for WSLv2 support, see the [original bug report](https://github.com/microsoft/WSL/issues/4518) and [this resulting discussion](https://github.com/microsoft/WSL/discussions/5855).

#### LocationParseError/InvalidURL: Failed to parse
This is usually seen when using USB and comes from the underlying libraries being unable to decipher the Link Local address used. This can be fixed by updating those libraries as follows.
```
$: python -m pip install --upgrade urllib3 requests
```

:::warning Windows
On Windows, there is a small chance that the specific IPv6 address used by the Moku can trigger a [bug in a core Python library](https://github.com/psf/requests/issues/6282), leading to an `InvalidURL` error. If the above steps don't help, the work-around is to replace the `%` character in the IP address with the sequence `%2525` (i.e. a doubly-escaped percent character). For example, if the original IP address ended in `023e%61`:

```python
o = Oscilloscope('[fe80:0000:0000:0000:7269:79ff:feb9:023e%252561]')
```
:::

#### IPv6 (including USB) Connection Issues
There are some environmental limitations when using IPv6, including using the Moku USB interface. See [this section](/ip-address.html#ipv6) for more information.


#### Access Requests.session
To configure any underlying session attributes, append `session_` to the actual attribute and pass it to the instrument constructor. For example, `trust_env` can be configured as `session_trust_env`.

```python
o = Oscilloscope("192.168.###.###", session_trust_env=False)
```