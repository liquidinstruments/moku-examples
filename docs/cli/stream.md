# mokucli stream

Stream LI binary data from Moku onto a local network port

## Usage

```console
$ mokucli stream [OPTIONS]
```

## Options

- `--ip-address TEXT`: IP address of the Moku
- `--stream-id TEXT`: Stream ID, this is part of `start_streaming` response
- `--target TEXT`: Target to write data to - port, file, or stdout
  - **PORT**: Local port number between 1024 and 65535
  - **FILE**: A valid filename with extension `.csv`, `.mat`, or `.npy`
  - **STDOUT**: Use `-` to print to console
- `--help`: Show this message and exit

## Examples

```bash
# Stream to TCP port 8005
mokucli stream --ip-address 192.168.1.100 --stream-id logsink0 --target 8005

# Stream to CSV file
mokucli stream --ip-address 192.168.1.100 --stream-id logsink0 --target stream_data.csv

# Stream to NumPy file
mokucli stream --ip-address 192.168.1.100 --stream-id logsink0 --target stream_data.npy

# Stream to MATLAB file
mokucli stream --ip-address 192.168.1.100 --stream-id logsink0 --target stream_data.mat

# Stream to standard output (console)
mokucli stream --ip-address 192.168.1.100 --stream-id logsink0 --target -
```

## Target Types

### Network Port
When targeting a port, the decoded stream data is made available on the specified TCP port. Client applications can connect to this port to receive the data.

### File Output
Files are created in the current working directory:
- `.csv` - Comma-separated values format
- `.mat` - MATLAB format
- `.npy` - NumPy binary format

### Standard Output
Using `-` as the target prints the converted stream data to the console.

## Use Cases

This command is commonly used:
- Inside client packages to receive Data Logger streams
- For languages without dedicated client packages
- To save streaming data directly to files
- For real-time data processing pipelines

## Notes

- The stream ID is obtained from the instrument's `start_streaming` API response
- The stream continues until interrupted with Ctrl+C
- File output automatically converts the binary stream to the specified format
- Network streaming provides raw decoded data for custom processing