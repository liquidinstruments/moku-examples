# mokucli files

Manage files on Moku devices

## Usage

```console
$ mokucli files [OPTIONS] COMMAND [ARGS]...
```

## Commands

- `list`: List files on the Moku
- `download`: Download files from the Moku
- `delete`: Delete files from the Moku

## mokucli files list

List files stored on the Moku device

### Usage

```console
$ mokucli files list [OPTIONS] IP_ADDRESS
```

### Arguments

- `IP_ADDRESS`: IP address of the Moku [required]

### Options

- `--name TEXT`: Filter to apply (supports wildcards)
- `--help`: Show this message and exit

### Examples

```bash
# List all files
mokucli files list 192.168.1.100

# List files containing "LockInAmplifier"
mokucli files list 192.168.1.100 --name "*LockInAmplifier*"

# List files with specific pattern
mokucli files list 192.168.1.100 --name "MokuDataLogger*2023*"
```

## mokucli files download

Download files from the Moku device

### Usage

```console
$ mokucli files download [OPTIONS] IP_ADDRESS
```

### Arguments

- `IP_ADDRESS`: IP address of the Moku [required]

### Options

- `--name TEXT`: Filter to apply (supports wildcards)
- `--help`: Show this message and exit

### Examples

```bash
# Download all files
mokucli files download 192.168.1.100

# Download files containing "LockInAmplifier"
mokucli files download 192.168.1.100 --name "*LockInAmplifier*"

# Download specific file pattern
mokucli files download 192.168.1.100 --name "MokuDataLogger*2023*"
```

## mokucli files delete

Delete files from the Moku device

### Usage

```console
$ mokucli files delete [OPTIONS] IP_ADDRESS
```

### Arguments

- `IP_ADDRESS`: IP address of the Moku [required]

### Options

- `--name TEXT`: Filter to apply (supports wildcards)
- `--help`: Show this message and exit

### Examples

```bash
# Delete all files (use with caution!)
mokucli files delete 192.168.1.100

# Delete files containing "LockInAmplifier"
mokucli files delete 192.168.1.100 --name "*LockInAmplifier*"

# Delete old files
mokucli files delete 192.168.1.100 --name "*2022*"
```

## Notes

- Files are typically data recordings from instruments (.li files)
- Files may not always be in the .li format, for instance, if you're deleting files from an SD card mounted to a Moku:Lab, there may be existing files on the SD card that are not in the .li format
- Downloaded files are saved to the current directory
- The `--name` filter supports standard wildcards (* and ?)
- Delete operations are permanent and cannot be undone