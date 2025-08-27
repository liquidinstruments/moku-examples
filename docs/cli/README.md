---
tags: ['CLI', 'Moku CLI', 'utilities', 'command line utility']
title: 'Moku CLI'
---

# Moku CLI

Moku CLI (`mokucli`) is a powerful command line utility for managing Moku hardware devices. It provides comprehensive tools for firmware updates, instrument bitstream management, feature deployment, and device administration.

## Installation

The latest packages are available to download from [Utilities](https://www.liquidinstruments.com/software/utilities/). The installation wizard will configure everything needed to launch the CLI successfully.

### Search Path Configuration

#### Windows and Mac
When using the installers, `mokucli` will be automatically discoverable from any client driver package (e.g., MATLAB or Python).

#### Linux
For Linux users, you must manually either:
1. Create a symbolic link to `mokucli` in the `/usr/local/bin` directory, or
2. Set the `MOKU_CLI_PATH` environment variable to the absolute path of `mokucli`

## Quick Start

Here are some common mokucli commands to get you started:

### Finding Devices
```bash
# List all Moku devices on your network
mokucli list
```

### Downloading the instruments for API Use
```bash
# Download all bitstreams for version 4.0.1 to local cache
mokucli instrument download 4.0.1
```

### Downloading Resources for Offline Use

```bash
# Download a specific bitstream
mokucli instrument download 4.0.1:oscilloscope --hw-version mokugo

# Download features for offline installation
mokucli feature download 4.0.1:api-server --hw-version mokupro
```

### Uploading Firmware
```bash
# Upload firmware version 4.0.1 to a device
mokucli firmware upload 192.168.1.100 4.0.1

# Upload a local firmware file
mokucli firmware upload 192.168.1.100 ./moku-mokupro-611.fw
```

### Managing Instruments on Device
```bash
# Upload a single instrument oscilloscope to a device
mokucli instrument upload 192.168.1.100 4.0.1:oscilloscope

# Upload multiple bitstreams, this uploads all downloaded single instruments
mokucli instrument upload 192.168.1.100 "4.0.1:01-*"
```

For more details on the supported syntax, see the [`mokucli instrument upload` documentation](./instrument.md#mokucli-instrument-upload)

## Command Overview

MokuCLI is organized into command groups, each serving a specific purpose:

1. **[instrument](instrument.md)** - Manage instrument bitstreams (download to cache, upload to device, list available)
2. **[firmware](firmware.md)** - Manage device firmware updates
3. **[feature](feature.md)** - Manage software features and plugins (download to cache, upload to device)
<!-- 4. **[command](command.md)** - Execute Moku API commands directly from CLI -->
5. **[convert](convert.md)** - Convert Liquid Instruments binary data files
6. **[download](download.md)** - Legacy bitstream download (use `instrument download`)
7. **[list](list.md)** - Search for Moku devices on network
8. **[files](files.md)** - Manage files on Moku devices
9. **[license](license.md)** - Manage device licenses
10. **[proxy](proxy.md)** - Create network proxy to device
11. **[stream](stream.md)** - Stream binary data from device

For advanced configuration options, see the [Advanced Usage](advanced.md) guide.

## Download and Install Workflow

MokuCLI follows a two-step workflow for managing resources (firmware, instruments, features such as rest-api):

1. **Download** - Resources are first downloaded to your local cache
2. **Install/Upload** - Resources are then installed from the local cache to Moku devices

This approach enables:
- **Offline installations** - Install resources without internet connectivity
- **Bulk deployments** - Deploy the same resources to multiple devices efficiently
- **API integration** - The Moku API can install instruments on-demand from the local cache
- **Faster reinstalls** - Resources are cached locally for quick access


## Getting Help

To see available commands and options:
```bash
mokucli --help
mokucli COMMAND --help
```

For detailed information about each command, see the individual command pages linked above.