# Advanced Usage

:::warning NOTE
These features will only be required in complex environments or for debugging an installation. To be used at the direction of the support team. [Contact support](https://www.liquidinstruments.com/support/contact/) for further information.
:::

This guide covers advanced MokuCLI features including configuration management, custom servers, and debugging options.

## Configuration Management

MokuCLI uses a configuration file to manage settings such as server definitions and data directories. The `config` command group provides tools to manage this configuration.

### mokucli config

Configuration file management commands

#### Usage

```console
$ mokucli config [OPTIONS] COMMAND [ARGS]...
```

#### Commands

- `create`: Create a default configuration file
- `which`: Print the default configuration file location
- `print`: Pretty-print the parsed configuration

### Configuration File Location

The configuration file is stored in platform-specific locations:

- **Linux**: `~/.config/moku/mokucli.yaml`
- **macOS**: `~/Library/Application Support/Moku/mokucli.yaml`
- **Windows**: `%APPDATA%\Moku\mokucli.yaml`

### Creating a Configuration File

```bash
# Create default configuration
mokucli config create

# Check configuration file location
mokucli config which

# View current configuration
mokucli config print
```

### Output

```bash
# Create and save default configuration
$ mokucli config create --save
✓ Configuration saved to: C:\Users\user\AppData\Roaming\Moku\mokucli.yaml

# Check configuration file location
$ mokucli config which
C:\Users\user\AppData\Roaming\Moku\mokucli.yaml

# View current configuration
$ mokucli config print
✓ Configuration from: C:\Users\user\AppData\Roaming\Moku\mokucli.yaml

  1 servers:
  2   default:
  3     type: http
  4     url: https://updates.liquidinstruments.com/static/
  5 data_dir: C:\Users\user\AppData\Roaming\Moku\data
  6
```

## Configuration File Format

The configuration file uses YAML format with the following structure:

```yaml
# Data directory for downloaded files
data_dir: /path/to/data/directory

# Server configurations
servers:
  # Default HTTP server (always present)
  default:
    type: http
    url: https://updates.liquidinstruments.com/static/
  
  # Example S3 server configuration
  my-s3-server:
    type: s3
    endpoint: s3.example.com
    bucket: moku-resources
    access_key: YOUR_ACCESS_KEY
    secret_key: YOUR_SECRET_KEY
    secure: true
```

### Server Configuration

Servers can be either HTTP or S3 type:

#### HTTP Server

```yaml
servers:
  ...
  server-name:
    type: http
    url: https://example.com/path/
```

#### S3 Server

```yaml
servers:
  ...
  server-name:
    type: s3
    endpoint: s3.example.com
    bucket: bucket-name
    access_key: ACCESS_KEY
    secret_key: SECRET_KEY
    secure: true  # Use HTTPS
```

## Using Custom Servers

Once servers are defined in the configuration file, they can be used with the `--server` option in various commands:

```bash
# Use custom server for downloads
mokucli instrument download 4.0.1 --server my-s3-server

# List resources from custom server
mokucli instrument list my-s3-server

# Upload firmware from custom server
mokucli firmware upload 192.168.1.100 4.0.1 --server my-s3-server
```

The `--server` option is available for:
- `instrument download`
- `instrument upload`
- `firmware upload`
- `feature download`
- `feature upload`

## Data Directory

Downloaded resources are cached in the data directory, which defaults to:
- **Linux**: `~/.config/moku/data/`
- **macOS**: `~/Library/Application Support/Moku/data/`
- **Windows**: `%APPDATA%\Moku\data\`

The directory structure is:
```
data/
├── firmware/
│   └── VERSION/
│       └── HARDWARE/
│           └── moku.fw
├── instruments/
│   └── VERSION/
│       └── HARDWARE/
│           └── *.bar
├── features/
│   └── FEATURE_NAME/
│       └── VERSION/
│           └── HARDWARE/
│               └── *.hgp
└── versions/
    └── *.json
```

## Debug Mode

Enable debug output by setting the `MOKUCLI_DEBUG` environment variable:

```bash
# Enable debug mode
export MOKUCLI_DEBUG=1
mokucli list

# Or inline
MOKUCLI_DEBUG=1 mokucli instrument download 4.0.1
```

Debug mode provides:
- Detailed server information
- S3 request details
- File path resolution information
- Additional error context

## Version Formats and Build Numbers

MokuCLI supports two version formats:

### Semantic Versions (Recommended)
Human-friendly versions like "4.0.1" that are used throughout the documentation. These versions:
- Are easier to remember and communicate
- Automatically resolve to appropriate build numbers
- Provide consistent versioning across all resource types

### Build Numbers (Advanced)
Direct numeric identifiers that represent specific builds:
- Firmware: e.g., "611", "612"
- Instruments: e.g., "18079", "18080"
- Features: Various numbers depending on the feature

Build numbers can be used anywhere a version is expected, but semantic versions are recommended for clarity. When using build numbers:

```bash
# Using build numbers directly
mokucli instrument download 18079 --hw-version mokugo
mokucli firmware upload 192.168.1.100 611
mokucli feature download 611/api-server
```

### Version Resolution
When you use a semantic version, MokuCLI:
1. Downloads a version mapping file (e.g., `versions/4.0.1.json`)
2. Extracts the appropriate build number for each resource type
3. Uses that build number for the actual operation

The mapping files are cached locally in the `versions/` directory for offline use.

## Best Practices

1. **Server Names**: Use descriptive names for custom servers (e.g., `dev`, `staging`, `prod`)

2. **Security**: Store sensitive credentials (access keys) securely:
   - Use environment variables for credentials
   - Restrict configuration file permissions
   - Consider using IAM roles for S3 access

3. **Data Management**: Periodically clean the data directory to free space:
   ```bash
   # View data directory location
   mokucli config print | grep data_dir
   
   # Remove old versions manually
   rm -rf ~/.config/moku/data/instruments/OLD_VERSION
   ```

4. **Version Resolution**: Semantic versions are resolved using JSON files in the `versions/` directory. These are automatically downloaded and cached when using semantic versions.

## Troubleshooting

### Configuration Issues

If configuration is not loading:
1. Check file exists: `mokucli config which`
2. Validate YAML syntax: `mokucli config print`
3. Ensure proper permissions on config file

### Server Connection Issues

For S3 servers:
- Verify endpoint URL is correct
- Check access credentials are valid
- Ensure bucket exists and is accessible
- Try with `secure: false` for non-HTTPS endpoints

### Debug Output

When reporting issues, include debug output:
```bash
MOKUCLI_DEBUG=1 mokucli [command] 2>&1 | tee mokucli-debug.log
```