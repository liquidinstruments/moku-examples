# mokucli firmware

Fetch and upload Moku firmware

## Usage

```console
$ mokucli firmware [OPTIONS] COMMAND [ARGS]...
```

## Commands

- `upload`: Upload Moku firmware updates
- `install`: Alias for upload

## mokucli firmware upload

Upload Moku firmware updates to a device

### Usage

```console
$ mokucli firmware upload [OPTIONS] IP_ADDRESS [FIRMWARE_REF]
```

### Arguments

- `IP_ADDRESS`: IP address or hostname of Moku (e.g., 192.168.1.100, MokuGo-000092) [required]
- `FIRMWARE_REF`: Firmware version or .fw file path [optional]

### Options

- `--download, -d`: Save the firmware file to the local cache (useful for offline or bulk installations)
- `--help`: Show this message and exit

### Examples

```bash
# Upload firmware version 4.0.1
mokucli firmware upload 192.168.1.100 4.0.1


# Upload and save firmware file to local cache
mokucli firmware upload 192.168.1.100 4.0.1 --download

# Upload from local firmware file
mokucli firmware upload 192.168.1.100 ./moku-mokupro-611.fw

# Upload latest firmware (when no version specified)
mokucli firmware upload 192.168.1.100
```

### Notes

- The device will show its current firmware version before uploading
- Firmware files are streamed directly from the server unless `--download` is specified
- When `--download` is used, the firmware is saved to the local cache for future use
- Downloaded firmware can be installed offline by referencing the local .fw file path
- After upload, the device will automatically install the firmware
- Status LEDs will flash during installation
- For Moku:Go, power cycle the device after the light turns off
- For other devices, they will automatically restart

### Installation Process

1. The firmware is uploaded to the device
2. The device validates the firmware file
3. Installation begins (status LEDs flash)
4. Device restarts automatically (except Moku:Go)

### Error Conditions

The upload may fail if:
- The firmware file is not found
- The firmware is not suitable for the device hardware
- Network connectivity issues occur
- The firmware file is corrupted