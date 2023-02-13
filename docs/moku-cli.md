---
---
# `Moku CLI`

## Introduction

Moku CLI (`mokucli`) is a command line utility to quickly access various features of the Moku hardware.

The latest packages are available to download from, [Software & Packages](https://www.liquidinstruments.com/resources/software/)

While the installation wizard configures everything required to launch the CLI successfully, in a few cases, customers are required to manually add an environment variable or update the $PATH variable based on the operating system. Setting `MOKU_CLI_PATH` with the absolute path to the installed CLI as an environment variable should fix such issues.

:::tip
Moku CLI is evolving and is subject to change, it is recommended to always install the latest package.
:::

**Usage**:

```console
$ mokucli [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--version`
* `--install-completion`: Install completion for the current shell.
* `--show-completion`: Show completion for the current shell, to copy it or customize the installation.
* `--help`: Show this message and exit.

**Commands**:

* `calibration`: Calibrate the Moku
* `convert`: Convert Liquid Instruments binary file to...
* `diagnostics`: Run diagnostics on the given Moku and report...
* `files`: List, download and delete files from the Moku
* `license`
* `list`: Search for the mokus on network and display...
* `proxy`: Run a proxy from local machine to the Moku
* `stream`: Stream the LI binary data from Moku onto a...

## `mokucli calibration`

Calibrate the Moku

**Usage**:

```console
$ mokucli calibration [OPTIONS] IP_ADDRESS
```

**Arguments**:

* `IP_ADDRESS`: IP Address of the Moku  [required]

**Options**:

* `--filter [ADC|DAC|PPSU|PMIC|ALL]`: Filter the calibration result  [default: ALL]
* `--update TEXT`: Update the calibration coefficients
* `--help`: Show this message and exit.

## `mokucli convert`

Convert Liquid Instruments binary file to other formats

**Usage**:

```console
$ mokucli convert [OPTIONS] SOURCE
```

**Arguments**:

* `SOURCE`: [required]

**Options**:

* `--format [csv|npy|mat]`: [default: csv]
* `--help`: Show this message and exit.

**Example**:
```bash
# Convert .li file to csv
$: mokucli convert MokuDataLoggerData_20230114_142326.li 
Writing "MokuDataLoggerData_20230114_142326.csv"...
[===========================================================================]

# Convert .li file to npy
$: mokucli convert MokuDataLoggerData_20230114_142326.li --format=npy
Writing "MokuDataLoggerData_20230114_142326.npy"...
[===========================================================================]
```

## `mokucli diagnostics`  <Badge text="Moku:Pro"/>

Run diagnostics on the given Moku and report the results

**Usage**:

```console
$ mokucli diagnostics [OPTIONS] IP_ADDRESS
```

**Arguments**:

* `IP_ADDRESS`: IP Address of the Moku  [required]

**Options**:

* `--results PATH`: Directory to save results to as a JSON  [default: .]
* `--help`: Show this message and exit.

## `mokucli files`

List, download and delete files from the Moku

**Usage**:

```console
$ mokucli files [OPTIONS] IP_ADDRESS
```

**Arguments**:

* `IP_ADDRESS`: IP Address of the Moku  [required]

**Options**:

* `--action [LIST|DOWNLOAD|DELETE]`: Action to perform  [default: LIST]
* `--name TEXT`: Filter to apply
* `--help`: Show this message and exit.

**Example**:
```bash
# list files on the moku
$: mokucli files 192.168.#.#
MokuLockInAmplifierData_20230207_071706.li
MokuDataLoggerData_20230206_155539.li

# download all files
$: mokucli files 192.168.#.# --action DOWNLOAD
Downloading MokuLockInAmplifierData_20230207_071706.li
[##############################] Done!
Downloading MokuDataLoggerData_20230206_155539.li
[##############################] Done!

# download all files where the name contains LockInAmplifier
$: mokucli files 192.168.#.# --action=download --name=*LockInAmp*
Downloading MokuLockInAmplifierData_20230207_071706.li
[##############################] Done!
```

## `mokucli license`

**Usage**:

```console
$ mokucli license [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `fetch`: Fetch the latest license file
* `list`: List available licenses
* `update`: Update license on the Moku

### `mokucli license fetch`

Fetch the latest license file

**Usage**:

```console
$ mokucli license fetch [OPTIONS] IP_ADDRESS
```

**Arguments**:

* `IP_ADDRESS`: IP Address of the Moku  [required]

**Options**:

* `--path PATH`: Directory to save the license file  [default: .]
* `--help`: Show this message and exit.

### `mokucli license list`

List available licenses

**Usage**:

```console
$ mokucli license list [OPTIONS] IP_ADDRESS
```

**Arguments**:

* `IP_ADDRESS`: IP Address of the Moku  [required]

**Options**:

* `--help`: Show this message and exit.

**Examples**:
```bash
$: mokucli license list 192.168.*.*
Entitlements
============
Oscilloscope
Lock-in Amplifier
Waveform Generator
FIR Filter Builder
```
### `mokucli license update`

Update license on the Moku

**Usage**:

```console
$ mokucli license update [OPTIONS] IP_ADDRESS
```

**Arguments**:

* `IP_ADDRESS`: IP Address of the Moku  [required]

**Options**:

* `--filename PATH`: Path to the license file
* `--help`: Show this message and exit.

## `mokucli list`

Search for the mokus on network and display the results

**Usage**:

```console
$ mokucli list [OPTIONS]
```

**Options**:

* `--help`: Show this message and exit.
  
**Example**:
```bash
$: mokucli list
Name                 Serial  HW     FW     IP                  
--------------------------------------------------------
MokuGo-000016        16      Go     576    10.1.111.145   
```

## `mokucli proxy`

Run a proxy from local machine to the Moku

**Usage**:

```console
$ mokucli proxy [OPTIONS] IP_ADDRESS
```

**Arguments**:

* `IP_ADDRESS`: IP Address of the Moku  [required]

**Options**:

* `--port INTEGER`: Local port, typically a number between 1024 and 65535 on which nothing else is running  [default: 8090]
* `--help`: Show this message and exit.

**Example**:
```bash
$: mokucli proxy 192.168.1.1
Running a proxy from 192.168.1.1 to localhost:8090
```

## `mokucli stream`

Stream the LI binary data from Moku onto a local network port

**Usage**:

```console
$ mokucli stream [OPTIONS]
```

**Options**:

* `--ip-address TEXT`: IP address of the Moku
* `--stream-id TEXT`: Stream ID, this is part of `start_streaming` response.
* `--target TEXT`: Target to write data to, port, file, stdout
    * PORT : Local port, typically a number between 1024 and 65535 on which nothing else is running
    * FILE: A valid file name with one of `[csv, mat, npy]` as extensions. File will always be created in the current working directory
    * STDOUT: Prints the converted stream to console
* `--help`: Show this message and exit.

**Example**:
```bash
# stream to TCP port 8005
$: mokucli --ip-address=192.168.1.1 --stream-id=logsink0 --target=8005
# stream to stream_data.csv file
$: mokucli --ip-address=192.168.1.1 --stream-id=logsink0 --target=stream_data.npy
# stream to standard out
$: mokucli --ip-address=192.168.1.1 --stream-id=logsink0 --target=-
```
