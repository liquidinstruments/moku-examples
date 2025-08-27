# mokucli stream

Stream LI binary data from Moku onto a local network port

## Usage

```console
$ mokucli stream [OPTIONS]
```

## Options

- `--ip-address TEXT`: IP address of the Moku
- `--stream-id TEXT`: Stream ID, this is part of `start_streaming` response
- `--target TEXT`: Target to write data to; port, file, or stdout (see [target types](#target-types) below)
  - PORT: Local port number between 1024 and 65535
  - FILE: A valid filename with extension `.csv`, `.mat`, or `.npy`
  - STDOUT: Use `-` to print to console
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

## Output

:::tip Note
The stream will continue for the duration set unless interrupted with `Ctrl+C`
:::

```bash
# Stream to standard output (console)
mokucli stream --ip-address 192.168.1.100 --stream-id logsink0 --target -
% Moku:Go Data Logger
% Input 1 (on), DC , 1 MOhm , 10 Vpp
% Input 2 (on), DC , 1 MOhm , 10 Vpp
% Acquisition rate: 1.0000000000e+02 Hz, Precision mode
% Output 1 (off) - Sine, 10.000 000 000 000 MHz, 1.000 0 Vpp, offset 0.000 0 V, phase 0.000 000 deg
% Output 2 (off) - Square, 50.000 000 000 kHz, 500.0 mVpp, offset 0.000 0 V, phase 0.000 000 deg, duty cycle 50.00 %
%
% Acquired 2025-08-27 T 12:01:15 +1000
% Time (s), Input 1 (V), Input 2 (V)
0.0000000000e+00, 1.6158858657e-02, -3.1586672866e-03
1.0000000000e-02, 1.6205819086e-02, -2.6268451131e-03
2.0000000000e-02, 1.6088761931e-02, -2.5640019285e-03
3.0000000000e-02, 1.6031327637e-02, -2.7892681696e-03
4.0000000000e-02, 1.6219919721e-02, -2.9075759359e-03
...
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
- The stream will continue for the duration set unless interrupted with Ctrl+C
- File output automatically converts the binary stream to the specified format
- Network streaming provides raw decoded data for custom processing