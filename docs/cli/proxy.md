# mokucli proxy

Run a proxy from local machine to Moku

## Usage

```console
$ mokucli proxy [OPTIONS] IP_ADDRESS
```

## Arguments

- `IP_ADDRESS`: IP address of the Moku [required]

## Options

- `--port INTEGER`: Local port, typically a number between 1024 and 65535 on which nothing else is running [default: 8090]
- `--help`: Show this message and exit

## Examples

```bash
# Run a proxy with default port
mokucli proxy 192.168.1.100

# Run a proxy on specific port
mokucli proxy 192.168.1.100 --port 8080
```

## Use Cases

The proxy command is particularly useful when:

1. **IPv6 to IPv4 Bridge**: The Moku is connected via IPv6 (e.g., USB connection) but you need IPv4 access
2. **Web Interface Access**: Accessing the Moku web interface when connected via USB
3. **Legacy Tool Support**: Using tools that don't support IPv6

## Example Workflow

1. Connect Moku via USB (which uses IPv6)
2. Run the proxy:
   ```bash
   mokucli proxy fe80::7269:79ff:feb0:1234%en0
   ```
3. Access the Moku web interface at: `http://localhost:8090`

## Output

```
Running a proxy from 192.168.1.100 to localhost:8090
```

The proxy will continue running until you stop it with Ctrl+C.

## Notes

- The proxied Moku is available on `localhost` using IPv4
- Make sure the chosen port is not already in use
- The proxy handles all HTTP traffic transparently
- Press Ctrl+C to stop the proxy