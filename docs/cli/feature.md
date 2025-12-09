# mokucli feature

Manage Moku features

## Usage

```console
$ mokucli feature [OPTIONS] COMMAND [ARGS]...
```

## Commands

-   `upload`: Upload software features to a Moku device
-   `list`: List available features on a server
-   `download`: Download features for a given version
-   `install`: Alias for upload

## mokucli feature upload

Upload software features to a Moku device

### Usage

```console
$ mokucli feature upload [OPTIONS] IP_ADDRESS FEATURES...
```

### Arguments

-   `IP_ADDRESS`: IP address or hostname of Moku (e.g., 192.168.1.100, MokuGo-000092) [required]
-   `FEATURES`: One or more features to upload, each either a path to a .hgp file, a version:feature identifier, or just a feature name (version taken from device). Supports versions (e.g., '4.0.1:api-server') and glob patterns: * (any characters), ? (single character), [seq] (character set), e.g., '4.0.1:api-*', '4.0.1:rest-?ttp', '4.0.1:[ar]*' [required]

### Options

-   `--verbose, -v`: Show detailed output for each feature upload
-   `--help`: Show this message and exit

### Examples

```bash
# Upload a single feature with explicit version
mokucli feature upload 192.168.1.100 4.0.1:api-server

# Upload a feature using device's version
mokucli feature upload 192.168.1.100 api-server

# Upload using hostname (device version auto-detected)
mokucli feature upload MokuGo-000123 api-server

# Upload multiple features using glob patterns
mokucli feature upload 192.168.1.100 "4.0.1:api-*"

# Upload a local feature file
mokucli feature upload 192.168.1.100 ./api-server_mokupro_611.hgp

# Upload with verbose output (shows version detection)
mokucli feature upload 192.168.1.100 api-server -v
```

## mokucli feature list

List available features on a server

### Usage

```console
$ mokucli feature list [OPTIONS] [SERVER]
```

### Arguments

-   `SERVER`: Server to list features from. Defaults to the default server. [default: None]

### Options

-   `--hw-version [mokugo|mokupro|mokulab|mokudelta]`: Filter by hardware version
-   `--version TEXT`: Filter by version (e.g., '4.0.1')
-   `--summary`: Show summary of available features without listing individual files
-   `--help`: Show this message and exit

### Examples

```bash
# List all available features
mokucli feature list

# List features for a specific version
mokucli feature list --version 4.0.1

# List features for a specific hardware version
mokucli feature list --hw-version mokupro

# Show summary without detailed listings
mokucli feature list --summary
```

### Output

```bash
# List all available features
$ mokucli feature list
ℹ Listing features...
ℹ Available features:

api-server:
  Version 619:
    moku20: api-server_moku20_619.hgp
    mokuaf: api-server_mokuaf_619.hgp
    mokugo: api-server_mokugo_619.hgp
    mokupro: api-server_mokupro_619.hgp
```

## mokucli feature download

Download features to the local cache for later installation

This command downloads software features to your local cache, enabling:

-   Offline installation to Moku devices
-   Bulk deployment of features across multiple devices
-   Pre-staging features before field deployment
-   Faster subsequent installations from local cache

Downloaded features are stored in the platform-specific data directory and can be installed later using `mokucli feature upload`.

### Usage

```console
$ mokucli feature download [OPTIONS] FEATURES...
```

### Arguments

-   `FEATURES`: One or more features to download (e.g., 'api-server', '4.0.1:api-server'). Format: version:feature or just feature name (requires --ip to get version from device) [required]

### Options

-   `--target PATH`: Directory to download features to [default: platform-specific data directory]
-   `--force`: Force redownload even if file exists
-   `--ip TEXT`: IP address of a connected Moku device (for version and hardware detection)
-   `--hw-version [mokugo|mokupro|mokulab|mokudelta]`: Hardware version to download. If not specified, downloads for all available hardware versions
-   `--verbose, -v`: Show detailed output for each feature download
-   `--help`: Show this message and exit

### Examples

```bash
# Download a specific feature with explicit version
mokucli feature download 4.0.1:api-server --hw-version mokupro

# Download using device's version (must specify --ip)
mokucli feature download api-server --ip 192.168.1.100

# Download using hostname instead of IP
mokucli feature download api-server --ip MokuGo-000123

# Download to specific directory
mokucli feature download 4.0.1:api-server --target ./features

# Force redownload
mokucli feature download 4.0.1:api-server --force

# Download for all hardware versions
mokucli feature download 4.0.1:api-server
```

### Notes

-   Features are software packages that enhance device functionality
-   The device hardware version is automatically detected when uploading
-   When no version is specified in the feature name, the device's current version is used automatically
-   If a feature is not found locally, it will be automatically downloaded before uploading
-   The device must be restarted after feature uploads to complete the update
-   Auto-download is not available for glob patterns - download them first using `feature download`
