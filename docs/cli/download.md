# mokucli download

Download bitstreams to the local cache for later installation

:::warning Legacy Command
This is a legacy command maintained for backward compatibility. Please use `mokucli instrument download` instead.
:::

This command downloads instrument bitstreams to your local cache for offline installation or bulk deployment. The downloaded bitstreams can be installed later using `mokucli upload` or `mokucli instrument upload`.

## Usage

```console
$ mokucli download [OPTIONS] VERSION_SPEC...
```

## Arguments

- `VERSION_SPEC`: One or more versions to download (e.g., 4.0.1), optionally suffixed with a specific bitstream identifier. Supports glob patterns: * (any characters), ? (single character), [seq] (character set), e.g., '4.0.1/01-*-00', '4.0.1/01-00?-*', '4.0.1/01-[0-9]*-00' [required]

## Options

- `--target PATH`: Directory to download bitstreams to [default: platform-specific data directory]
- `--force / --no-force`: Force rewrite by ignoring checksum [default: no-force]
- `--ip TEXT`: IP address of a connected Moku device (for hardware version detection)
- `--hw-version [mokugo|mokupro|mokulab|mokudelta]`: Hardware version to use when no device is connected
- `--verbose, -v`: Show detailed output for each bitstream download
- `--help`: Show this message and exit

## Examples

```bash
# Download all bitstreams for version 4.0.1
mokucli download 4.0.1

# Download specific bitstream
mokucli download 4.0.1/oscilloscope --hw-version mokugo

# Download to specific directory
mokucli download 4.0.1 --target ./bitstreams
```

## Migration Guide

To migrate from `mokucli download` to `mokucli instrument download`, simply add `instrument` after `mokucli`:

```bash
# Old command
mokucli download 4.0.1

# New command
mokucli instrument download 4.0.1
```

All options and arguments remain the same.