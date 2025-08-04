# mokucli list

Search for Moku devices on network and display results

## Usage

```console
$ mokucli list [OPTIONS]
```

## Options

- `--help`: Show this message and exit

## Examples

```bash
# List all Moku devices on your network
mokucli list
```

## Output

The command displays a table with the following information for each discovered device:

- **Name**: Device hostname
- **Serial**: Device serial number
- **HW**: Hardware version (Go, Pro, Lab, Delta)
- **FW**: Firmware version
- **IP**: IP address

Example output:
```
Name                 Serial  HW     FW     IP
--------------------------------------------------------
MokuGo-000016        16      Go     576    10.1.111.145
MokuPro-000123       123     Pro    611    192.168.1.100
```

## Notes

- Device discovery uses mDNS/Zeroconf protocol
- Devices must be on the same network segment as your computer
- Discovery may take a few seconds
- Firewall settings may block device discovery