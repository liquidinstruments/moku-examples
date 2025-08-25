# mokucli instrument

Manage Moku instrument bitstreams

## Usage

```console
$ mokucli instrument [OPTIONS] COMMAND [ARGS]...
```

## Commands

- `list`: List available bitstreams on a server
- `download`: Download bitstreams for a given version
- `upload`: Upload a bitstream to a Moku device
- `install`: Alias for upload

## mokucli instrument list

List available bitstreams on a server

### Usage

```console
$ mokucli instrument list [OPTIONS] [SERVER]
```

### Arguments

- `SERVER`: Server to list bitstreams from. Defaults to the default server. [default: None]

### Options

- `--hw-version [mokugo|mokupro|mokulab|mokudelta]`: Filter by hardware version
- `--version TEXT`: Filter by version (e.g., 4.0.1)
- `--summary`: Show summary of available bitstreams without listing individual files
- `--help`: Show this message and exit

### Examples

```bash
# List all available bitstreams
mokucli instrument list

# List bitstreams for a specific version
mokucli instrument list --version 4.0.1

# List bitstreams for a specific hardware version
mokucli instrument list --hw-version mokugo

# Show summary without detailed listings
mokucli instrument list --summary
```

## mokucli instrument download

Download bitstreams to the local cache for later installation

This command downloads instrument bitstreams to your local cache, allowing for:
- Offline installation to Moku devices
- Bulk deployment across multiple devices
- Use with the Moku API for on-demand instrument installation
- Pre-staging instruments before field deployment

Downloaded bitstreams are stored in the platform-specific data directory and can be installed later using `mokucli instrument upload`.

### Usage

```console
$ mokucli instrument download [OPTIONS] VERSION_SPEC...
```

### Arguments

- `VERSION_SPEC`: One or more versions to download (e.g., 4.0.1), optionally suffixed with a specific bitstream identifier using colon separator. Supports glob patterns: * (any characters), ? (single character), [seq] (character set), e.g., '4.0.1:01-*-00', '4.0.1:01-00?-*', '4.0.1:01-[0-9]*-00' [required]

### Options

- `--target PATH`: Directory to download bitstreams to [default: platform-specific data directory]
- `--force / --no-force`: Force redownload even if file exists [default: no-force]
- `--ip TEXT`: IP address of a connected Moku device (for hardware version detection and auto-version detection)
- `--hw-version [mokugo|mokupro|mokulab|mokudelta]`: Hardware version to use when no device is connected. When --ip is provided without a version, the firmware version is auto-detected from the connected device
- `--verbose, -v`: Show detailed output for each bitstream download
- `--help`: Show this message and exit

### Examples

```bash
# Download all bitstreams for version 4.0.1
mokucli instrument download 4.0.1

# Download specific bitstream
mokucli instrument download 4.0.1:oscilloscope --hw-version mokugo

# Download multiple bitstreams using glob patterns
mokucli instrument download "4.0.1:01-*" --hw-version mokupro

# Download to specific directory
mokucli instrument download 4.0.1 --target ./bitstreams

# Download with device auto-detection
mokucli instrument download 4.0.1 --ip 192.168.1.100

# Force redownload
mokucli instrument download 4.0.1:oscilloscope --force
```

## mokucli instrument upload

Upload bitstreams to a Moku device

### Usage

```console
$ mokucli instrument upload [OPTIONS] IP_ADDRESS BITSTREAM...
```

### Arguments

- `IP_ADDRESS`: IP address or hostname of Moku (e.g., 192.168.1.100, MokuGo-000092) [required]
- `BITSTREAM`: One or more bitstreams to upload, each either a path to a bitstream file or a version:instrument identifier. Supports glob patterns: * (any characters), ? (single character), [seq] (character set), e.g., '4.0.1:01-*-00', '4.0.1:01-00?-*', '4.0.1:01-[0-9]*-00' [required]

### Options

- `--verbose, -v`: Show detailed output for each bitstream upload
- `--help`: Show this message and exit

### Examples

```bash
# Upload a single bitstream
mokucli instrument upload 192.168.1.100 4.0.1:oscilloscope

# Upload multiple bitstreams using glob patterns
mokucli instrument upload 192.168.1.100 "4.0.1:01-*"

# Upload a local bitstream file
mokucli instrument upload 192.168.1.100 ./oscilloscope.bar

# Upload with verbose output
mokucli instrument upload 192.168.1.100 4.0.1:oscilloscope -v
```

### Notes

- The device hardware version is automatically detected when uploading
- If a bitstream is not found locally, it will be automatically downloaded before uploading
- The device's bitstream cache is automatically reloaded after successful uploads
- Auto-download is not available for glob patterns - download them first using `instrument download`