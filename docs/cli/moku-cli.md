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
-   `files`: List, download and delete files from the Moku
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

-   `SOURCE`: \[required\]

### Options

-   `--help`: Show this message and exit.

### Examples

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

## mokucli download

Download bitstreams for a given firmware version

### Usage

```console
$ mokucli download [OPTIONS] FW_VER
```

### Arguments

-   `FW_VER`: \[required\]

### Options

-   `--help`: Show this message and exit.
-   `target PATH`: File path to download bitstreams to  \[default: .\]
-   `force / --no-force`: Force rewrite by ignoring checksum  \[default: no-force\]

### Examples

```bash
# download bitstreams for firmware version 600
$: mokucli download 600
Downloading latest instruments for firmware version 600...
[===========================================================================]
```

## mokucli files

List, download and delete files from the Moku

### Usage

```console
$ mokucli files [OPTIONS] IP_ADDRESS
```

### Arguments

-   `IP_ADDRESS`: IP Address of the Moku \[required\]

### Options

-   `--action [LIST|DOWNLOAD|DELETE]`: Action to perform \[default: LIST\]
-   `--name TEXT`: Filter to apply
-   `--help`: Show this message and exit.

### Examples

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
$: mokucli license list 192.168.*.*
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

## mokucli list

Search for the mokus on network and display the results

### Usage

```console
$ mokucli list [OPTIONS]
```

### Options

-   `--help`: Show this message and exit.

### Examples

```bash
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
$: mokucli proxy 192.168.1.1
Running a proxy from 192.168.1.1 to localhost:8090
```

## mokucli stream

Stream the LI binary data from Moku onto a local network port, `stdout`, or to a file. This is commonly used inside client packages, or for languages without dedicated client packages, to receive Data Logger streams and make the decoded data available for easy consumption.

### Usage

```console
$ mokucli stream [OPTIONS]
```

### options

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
$: mokucli --ip-address=192.168.1.1 --stream-id=logsink0 --target=8005
# stream to stream_data.csv file
$: mokucli --ip-address=192.168.1.1 --stream-id=logsink0 --target=stream_data.npy
# stream to standard out
$: mokucli --ip-address=192.168.1.1 --stream-id=logsink0 --target=-
```
