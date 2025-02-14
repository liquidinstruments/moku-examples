# Moku CLI Usage

```console
$ mokucli [OPTIONS] COMMAND [ARGS]...
```

## Options

-   `--version` : Display the version and exit
-   `--install-completion`: Install auto completion for the current shell.
-   `--show-completion`: Show completion for the current shell, to copy it or customize the installation.
-   `--help`: Show this message and exit.

## Commands

-   `convert`: Convert Liquid Instruments binary file to CSV, NPY, MAT and HDF5
-   `download`: Download bitstreams for a given firmware version
-   `feature`: Check, install and upload Moku features
-   `files`: List, download and delete files from the Moku
-   `firmware`: Fetch and upload Moku firmware
-   `license`: List, fetch and reload the license entitlements
-   `list`: Search for the mokus on network
-   `proxy`: Run a proxy from local machine to the Moku
-   `stream`: Stream the LI binary data from Moku onto network port or a file

## mokucli convert

Convert Liquid Instruments binary file to other formats

### Usage

```console
$ mokucli convert [OPTIONS] SOURCE
```

### Arguments

-   `SOURCE`: \[required\] Full path, relative path or filename of the .li file to convert

### Options

-   `--format [csv|npy|mat|hdf5]`: File format type to convert .li to \[default: csv\]
-   `--help`: Show this message and exit.

### Examples

```bash
# Convert .li file to csv
$: mokucli convert MokuDataLoggerData_20230114_142326.li
Writing "MokuDataLoggerData_20230114_142326.csv"...
[===========================================================================]

# Convert .li file to npy
$: mokucli convert MokuDataLoggerData_20230114_142326.li --format npy
Writing "MokuDataLoggerData_20230114_142326.npy"...
[===========================================================================]
```

## mokucli download

Download bitstreams for a given firmware version

### Usage

```console
$ mokucli download [OPTIONS] FW_VER
```

### Arguments

-   `FW_VER`: Firmware version to download \[required\]

### Options

-   `--target PATH`: File path to download bitstreams to  \[default: .\]
-   `--force / --no-force`: Force rewrite by ignoring checksum  \[default: no-force\]
-   `--help`: Show this message and exit.

### Examples

```bash
# download bitstreams for firmware version 600
$: mokucli download 600
Downloading latest instruments for firmware version 600...
[===========================================================================]

# download bitstreams for firmware version 600 to target
$: mokucli download 600 --target ".\site-packages\moku\data"
Downloading latest instruments for firmware version 600...
[===========================================================================]
```

### Finding the target path Python

Find the path of your moku installation by looking at the location,
in the example below this is the line:
`Location: C:\Users\venv\Lib\site-packages` with `\moku\data` appended.
You may need to manually create the data folder before downloading the
bitstreams to this path.

```bash
# finding the target path for moku python installation
$: pip show moku
Name: moku
Version: 3.3.3
Summary: Python scripting interface to the Liquid Instruments Moku hardware
Home-page: https://liquidinstruments.com
Author: Liquid Instruments
Author-email: info@liquidinstruments.com
License: MIT
Location: C:\Users\venv\Lib\site-packages
Requires: requests
Required-by:

# check for or create the data folder
$: mkdir C:\Users\venv\Lib\site-packages\moku\data

# download bitstreams for firmware version 600 to target from location
$: mokucli download 600 --target "C:\Users\venv\Lib\site-packages\moku\data"
Downloading latest instruments for firmware version 600...
[===========================================================================]
```

## mokucli feature

Check, install and upload Moku features

::: warning Beta
This functionality is currently in beta and subject to change.
:::

### Usage

```console
mokucli feature [OPTIONS] COMMAND [ARGS]...
```

### Commands

-   `check`: Check for new feature updates.
-   `install`: Install an feature to a moku
-   `upload`: Upload a software feature

### Options

-   `--help`: Show this message and exit.

### mokucli feature check

Check for new feature updates.

#### Usage

```console
mokucli feature check [OPTIONS]
```

#### Options

-   `-i, --ip IP_ADDR`: IP Address of the Moku
-   `-f, --firmware, --fw TEXT`: Firmware version to check
-   `-h, --hardware, --hw [mokugo|mokupro|mokulab]`: Hardware version to check
-   `-o, --offline`: Don't check for feature updates from the server, just get Moku versions. Requires --ip.
-   `-H, --patch-host URL`: The HTTP server where index.json and feature updates can be found  \[default: <http://updates.liquidinstruments.com/static/patches/>\]
-   `--help`: Show this message and exit.

#### Examples

```bash
# check feature and firmware on a Moku device
$: mokucli feature check --ip 192.168.1.1 --offline
Checking for updates for Moku-Pro:

Currently installed packages:
    api-server: 3.3.2.1
    mercury: 601.0

# check for available feature and firmware updates for a Moku device
$: mokucli feature check --ip 192.168.1.1
Checking for updates for Moku-Pro:

Currently installed packages:
    api-server: 3.3.2.1
    mercury: 601.0

No feature updates found

# check available feature and firmware updates for hardware and firmware
$: mokucli feature check -h mokupro -f 601
Fetching updates for firmware mokupro-601

Feature updates found:
    api-server:  3.3.2.1
    Notes:
         * Fix Oscilloscope frontend configuration on a running instrument
         * Change handling of longer Oscilloscope data frames to only contain real data
         * Make get_data timeout handling consistent throughout the communication stack
```

### mokucli feature install

Install an feature to a moku

#### Usage

```console
mokucli feature install [OPTIONS] IP_ADDRESS PACKAGE
```

#### Arguments

-   `IP_ADDRESS`: IP Address of the Moku  \[required\]
-   `PACKAGE`: May be the name of a feature, e.g 'rest-http' to fetch from the server or the path to a file on disk 'rest-http-mokugo-3.3.1.2.hgp'  \[required\]

#### Options

-   `-n, --dry-run`: Don't install the feature on the device (use with --download to just fetch feature files)
-   `-d, --download`: Download the feature updates locally for offline installation.
-   `-H, --patch-host URL`: The HTTP server where index.json and feature updates can be found  \[default: <http://updates.liquidinstruments.com/static/patches/>\]
-   `--help`: Show this message and exit.

#### Examples

```bash
# upload and install software feature to your Moku device
$: mokucli feature install 192.168.1.1 api-server
Found api-server version  3.3.2.1
Notes:
     * Fix Oscilloscope frontend configuration on a running instrument
     * Change handling of longer Oscilloscope data frames to only contain real data
     * Make get_data timeout handling consistent throughout the communication stack

Uploading api-server_mokupro_601.hgp..... Done.
Please restart the Moku to complete the update process

# download a software feature locally
$: mokucli feature install --dry-run --download 192.168.1.1 api-server
Found api-server version  3.3.2.1
Notes:
     * Fix Oscilloscope frontend configuration on a running instrument
     * Change handling of longer Oscilloscope data frames to only contain real data
     * Make get_data timeout handling consistent throughout the communication stack

Patch written to: api-server_mokupro_601.hgp

To install, use:

    $ mokucli feature install 192.168.1.1 api-server_mokupro_601.hgp

# upload and install a local software feature to your Moku device
$: mokucli feature install 192.168.1.1 api-server_mokupro_601.hgp
Uploading api-server_mokupro_601.hgp..... Done.
Please restart the Moku to complete the update process
```

### mokucli feature upload

Upload a software feature

#### Usage

```console
mokucli feature upload [OPTIONS] IP_ADDRESS PATH
```

#### Arguments

-   `IP_ADDRESS`: IP Address of the Moku  \[required\]
-   `PATH`: Path to the .hgp file  \[required\]

#### Options

-   `--help`: Show this message and exit.

#### Examples

```bash
# upload a software feature to your Moku device
$: mokucli feature upload 192.168.1.1 api-server_mokupro_601.hgp
Uploading api-server_mokupro_601.hgp..... Done.
Please restart the Moku to complete the update process
```

## mokucli files

List, download and delete files from the Moku

### Usage

```console
$ mokucli files [OPTIONS] COMMAND [ARGS]...
```

### Arguments

-   `IP_ADDRESS`: IP Address of the Moku \[required\]

### Commands

-   `delete`: Delete files from the Moku
-   `download`: Download files from the Moku
-   `list`: List files from the Moku

### Options

-   `--help`: Show this message and exit.

### mokucli files delete

Delete files from the Moku

#### Usage

```console
mokucli files delete [OPTIONS] IP_ADDRESS
```

#### Arguments

-   `IP_ADDRESS`: IP Address of the Moku \[required\]

#### Options

-   `--name TEXT`: Filter to apply
-   `--help`: Show this message and exit.

#### Examples

```bash
# delete all files
$: mokucli files delete 192.168.1.1
Deleted file MokuLockInAmplifierData_20230207_071706.li
Deleted file MokuDataLoggerData_20230206_155539.li

# delete all files where the name contains LockInAmplifier
$: mokucli files delete 192.168.1.1 --name "*LockInAmplifier*"
Deleted file MokuLockInAmplifierData_20230207_071706.li
```

### mokucli files download

Download files from the Moku

#### Usage

```console
mokucli files download [OPTIONS] IP_ADDRESS
```

#### Arguments

-   `IP_ADDRESS`: IP Address of the Moku \[required\]

#### Options

-   `--name TEXT`: Filter to apply
-   `--help`: Show this message and exit.

#### Examples

```bash
# download all files
$: mokucli files download 192.168.1.1
Downloading MokuLockInAmplifierData_20230207_071706.li
[##############################] Done!
Downloading MokuDataLoggerData_20230206_155539.li
[##############################] Done!

# download all files where the name contains LockInAmplifier
$: mokucli files download 192.168.1.1 --name "*LockInAmplifier*"
Downloading MokuLockInAmplifierData_20230207_071706.li
[##############################] Done!
```

### mokucli files list

List files from the Moku

#### Usage

```console
mokucli files list [OPTIONS] IP_ADDRESS
```

#### Arguments

-   `IP_ADDRESS`: IP Address of the Moku \[required\]

#### Options

-   `--name TEXT`: Filter to apply
-   `--help`: Show this message and exit.

#### Examples

```bash
# list all files
$: mokucli files list 192.168.1.1
MokuLockInAmplifierData_20230207_071706.li
MokuDataLoggerData_20230206_155539.li

# list all files where the name contains LockInAmplifier
$: mokucli files list 192.168.1.1 --name "*LockInAmplifier*"
MokuLockInAmplifierData_20230207_071706.li
```

## mokucli firmware

Fetch and upload Moku firmware

### Usage

```console
mokucli firmware [OPTIONS] COMMAND [ARGS]...
```
  
### Options

-   `--help`: Show this message and exit.

### Commands

-   `install`: Install Moku firmware updates

### mokucli firmware install

Install Moku firmware updates

#### Usage

```console
mokucli firmware install [OPTIONS] IP_ADDRESS [FIRMWARE_REF]
```

#### Arguments

-   `IP_ADDRESS`: IP Address of the Moku  \[required\]
-   `FIRMWARE_REF`: Firmware version or .fw file path

#### Options

-   `-d, --download`: Save the firmware file to disk
-   `-w, --wait SECONDS`: Wait the given number of seconds for the device to reboot. Exits with a non-zero exit code if the device is not seen within that time.  \[default: -1\]
-   `-H, --patch-host URL`: The HTTP server where index.json and feature updates can be found  \[default: <http://updates.liquidinstruments.com/static/patches/>\]
-   `--help`: Show this message and exit.

#### Examples

```bash
# update the firmware on the Moku device
$: mokucli firmware install 192.168.1.1 601
You are currently running firmware 600
Found firmware versions:
601 - 2024-12-16
    Bug fixes and stability improvements

Installing firmware version 601
Uploading moku.fw............................................ Done.
Waiting for firmware to install..

# update the firmware on the Moku device and download the firmware file locally
$: mokucli firmware install --download 192.168.1.1 601
You are currently running firmware 600
Found firmware versions:
601 - 2024-12-16
    Bug fixes and stability improvements

Installing firmware version 601
File saved to: ./moku-mokupro-601.fw
Uploading moku.fw............................................ Done.
Waiting for firmware to install..

# update the firmware on the Moku device and from the local firmware file
$: mokucli firmware install 192.168.1.1 moku-mokupro-601.fw
Uploading moku.fw............................................ Done.
Waiting for firmware to install..
```

## mokucli license

::: warning Beta
This functionality is currently in beta and subject to change.
:::

### Usage

```console
$ mokucli license [OPTIONS] COMMAND [ARGS]...
```

### Options

-   `--help`: Show this message and exit.

### Commands

-   `fetch`: Fetch the latest license file from the Liquid Instruments license server and save it locally
-   `list`: List available licenses for a Moku
-   `update`: Update license on the Moku with one stored locally

### mokucli license fetch

Fetch the latest license file for a Moku device from the Liquid Instruments license server and save locally.

#### Usage

```console
$ mokucli license fetch [OPTIONS] IP_ADDRESS
```

#### Arguments

-   `IP_ADDRESS`: IP Address of the Moku \[required\]

#### Options

-   `--path PATH`: Directory to save the license file \[default: .\]
-   `--help`: Show this message and exit.

#### Examples

```bash
# save license file to local directory
$: mokucli license fetch 192.168.1.1
Saving license file to ./moku-0.0-000.lic

# save license file to specified directory
$: mokucli license fetch --path ./licenses 192.168.1.1
Saving license file to ./licenses/moku-0.0-000.lic
```

### mokucli license list

List available licenses for the given Moku device.

#### Usage

```console
$ mokucli license list [OPTIONS] IP_ADDRESS
```

#### Arguments

-   `IP_ADDRESS`: IP Address of the Moku \[required\]

#### Options

-   `--help`: Show this message and exit.

#### Examples

```bash
# list available instruments for the Moku device
$: mokucli license list 192.168.1.1
Entitlements
============
Oscilloscope
Lock-in Amplifier
Waveform Generator
FIR Filter Builder
```

### mokucli license update

Update license on the Moku with the latest from the Liquid Instruments license server, or one stored locally. The local file has typically been retrieved using a previous `moku license fetch`.

#### Usage

```console
$ mokucli license update [OPTIONS] IP_ADDRESS
```

#### Arguments

-   `IP_ADDRESS`: IP Address of the Moku \[required\]

#### Options

-   `--filename PATH`: Path to the license file. If no file is given, the license server is queried.
-   `--help`: Show this message and exit.

#### Examples

```bash
# update license on the Moku device from the license server
$: mokucli license update 192.168.1.1
Reloaded license on Moku 192.168.1.1

# update license on the Moku device from the local license file
$: mokucli license update --filename moku-0.0-000.lic 192.168.1.1
Uploaded license ./moku-0.0-000.lic to Moku 192.168.1.1
```

## mokucli list

Search for the Moku devices on network and display the results

### Usage

```console
$ mokucli list [OPTIONS]
```

### Options

-   `--help`: Show this message and exit.

### Examples

```bash
# list the Moku devices on your network
$: mokucli list
Name                 Serial  HW     FW     IP
--------------------------------------------------------
MokuGo-000016        16      Go     576    10.1.111.145
```

## mokucli proxy

Run a proxy from local machine to the Moku. The proxied Moku is available on `localhost` using IPv4. This is useful when the actual connection to the Moku uses IPv6 but you wish to use a tool without native IPv6 support. A common example is accessing the Moku web interface when the Moku is connected over USB.

### Usage

```console
$ mokucli proxy [OPTIONS] IP_ADDRESS
```

### Arguments

-   `IP_ADDRESS`: IP Address of the Moku \[required\]

### Options

-   `--port INTEGER`: Local port, typically a number between 1024 and 65535 on which nothing else is running \[default: 8090\]
-   `--help`: Show this message and exit.

### Example

```bash
# run a proxy from your local machine to your Moku device
$: mokucli proxy 192.168.1.1
Running a proxy from 192.168.1.1 to localhost:8090
```

## mokucli stream

Stream the LI binary data from Moku onto a local network port, `stdout`, or to a file. This is commonly used inside client packages, or for languages without dedicated client packages, to receive Data Logger streams and make the decoded data available for easy consumption.

### Usage

```console
$ mokucli stream [OPTIONS]
```

### Options

-   `--ip-address TEXT`: IP address of the Moku
-   `--stream-id TEXT`: Stream ID, this is part of `start_streaming` response.
-   `--target TEXT`: Target to write data to, port, file, stdout
    -   PORT : Local port, typically a number between 1024 and 65535 on which nothing else is running
    -   FILE: A valid file name with one of `[csv, mat, npy]` as extensions. File will always be created in the current working directory.
    -   STDOUT: Prints the converted stream to console
-   `--help`: Show this message and exit.

### Examples

```bash
# stream to TCP port 8005
$: mokucli --ip-address 192.168.1.1 --stream-id logsink0 --target 8005

# stream to stream_data.csv file
$: mokucli --ip-address 192.168.1.1 --stream-id logsink0 --target stream_data.npy

# stream to standard out
$: mokucli --ip-address 192.168.1.1 --stream-id logsink0 --target -
```
