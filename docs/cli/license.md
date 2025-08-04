# mokucli license

Manage Moku licenses

## Usage

```console
$ mokucli license [OPTIONS] COMMAND [ARGS]...
```

## Commands

- `list`: List available licenses for a Moku
- `fetch`: Fetch the latest license file from the Liquid Instruments license server
- `update`: Update license on the Moku

## mokucli license list

List available licenses/entitlements for the given Moku device

### Usage

```console
$ mokucli license list [OPTIONS] IP_ADDRESS
```

### Arguments

- `IP_ADDRESS`: IP address of the Moku [required]

### Options

- `--help`: Show this message and exit

### Examples

```bash
# List available instruments for the Moku device
mokucli license list 192.168.1.100
```

### Output Example

```
Entitlements
============
Oscilloscope
Lock-in Amplifier
Waveform Generator
FIR Filter Builder
PID Controller
Spectrum Analyzer
```

## mokucli license fetch

Fetch the latest license file for a Moku device from the Liquid Instruments license server and save locally

### Usage

```console
$ mokucli license fetch [OPTIONS] IP_ADDRESS
```

### Arguments

- `IP_ADDRESS`: IP address of the Moku [required]

### Options

- `--path PATH`: Directory to save the license file [default: current directory]
- `--help`: Show this message and exit

### Examples

```bash
# Save license file to current directory
mokucli license fetch 192.168.1.100

# Save license file to specific directory
mokucli license fetch 192.168.1.100 --path ./licenses
```

## mokucli license update

Update license on the Moku with the latest from the Liquid Instruments license server, or one stored locally

### Usage

```console
$ mokucli license update [OPTIONS] IP_ADDRESS
```

### Arguments

- `IP_ADDRESS`: IP address of the Moku [required]

### Options

- `--filename PATH`: Path to the license file. If no file is given, the license server is queried
- `--help`: Show this message and exit

### Examples

```bash
# Update license from the license server
mokucli license update 192.168.1.100

# Update license from local file
mokucli license update 192.168.1.100 --filename moku-4.0-000123.lic
```

## Notes

- License files are named in the format: `moku-{hw_version}-{serial}.lic`
- The `fetch` command is useful for offline license updates
- License updates may enable new instruments or features
- Internet connection is required when fetching from the license server