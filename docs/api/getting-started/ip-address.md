# Finding Your IP Address

There are many ways to find your Moku's IP Address. You may use the [Desktop App](https://liquidinstruments.com/products/download/) (if installed) or the [Moku CLI]((../../cli/)) tool ([Utilities](https://www.liquidinstruments.com/software/utilities/)). Additionally, if you're using the USB Connection to your Moku, the IP Address can be calculated based on the serial number.

## Using the Desktop App

-   Open the Desktop Application
-   On the "Select Device" screen, find the tile of the Moku to which you wish to connect
-   Right-click and select `Device Info > Copy IP Address`

## Moku CLI

<!-- :::tip Python Only
The `moku` Command Line Utility is installed as part of the Moku Scripting API for Python. If you're using MATLAB then you may choose to install the Python library additionally, or use one of the other methods listed on this page
::: -->
Install [mokucli](../../cli/) from our [Utilities](https://www.liquidinstruments.com/software/utilities/) page. At a terminal or command prompt, use the Moku CLI tool to list all the Moku devices your computer can see.

```
$: mokucli list
```

## USB Connection

API support over the USB is possible as the USB link presents as an Ethernet interface. The interface is configured to have an IPv6 Link-Local address only, which can be found using any of the methods above. IPv6 support is limited in some environments, see below.

<!-- Your Moku uses IPv6 over USB, it will have a pre-determined IPv6 address based on its serial number. You can find this address by entering your serial number in to the calculator below.

The Serial Number on a Moku:Go is found in three places:
- The six digits after the underscore on the top line of the barcode sticker underneath your Moku:Go
- The digits after the hyphen in the device's WiFi Access Point SSID
- In the `Device Info` drop-down when you right-click a Device Tile in "Select Device" screen of the Moku Desktop Application

<ip-calculator/> -->

## IPv6

:::danger IPv6 Support
IPv6, and especially IPv6 link-local addressing as used when your Moku is connected over USB, is not universally supported. As such the API cannot be used over USB in some environments without a proxy (see below). A non-exhaustive list of limited of **unsupported** environments is

-   Windows Subsystem for Linux Version 2 (WSL2), as discussed with workarounds [here](https://github.com/microsoft/WSL/discussions/5855)
-   LabVIEW, as discussed [here](https://forums.ni.com/t5/LabVIEW-Idea-Exchange/Native-support-for-IPv6/idi-p/1481942)
-   Most [web browsers](https://support.mozilla.org/en-US/questions/1111992) (which aren't full API clients but which may be required to use Moku Compile and other features)

If you require API connectivity from these environments, you must use a proxy as below, or a network connection like Ethernet or WiFi configured to use IPv4. Ethernet may be configured as a point-to-point network with Static IPs if security is a concern.
:::

### Specifying IPv6 Addresses

The above unsupported environments notwithstanding, some IPv6 errors simply come from how the address is specified. One should keep the following points in mind:

1. Most importantly **make sure your libraries are up to date**. In particular, Python parsing of these addresses was broken in some versions of `urllib3`. See [this Troubleshooting tip](./starting-python.md#ipv6-including-usb-connection-issues) for details.
2. Enclose the IP address in square brackets: `[fe80::1:2:3:4%eth0]`
3. If specifying the IP address on the command line, ensure those brackets are escaped if required, e.g. using single quotes in Bash: `'[fe80::1:2:3:4%eth0]'`. Windows CMD does not require escaping the brackets, and in fact will fail if you specify quotes (which will incorrectly be interpreted as part of the address)
4. If the scope id (element after the percent sign) is numeric, e.g. on Windows, ensure that your programming language does not interpret it as a special character

#### Examples

<code-group>
<code-block title="Python">

```python
i = Oscilloscope('[fe80::1:2:3:4%eth0]')
```

</code-block>

<code-block title="Bash">
```bash
$: curl 'https://[fe80::7269:79ff:feb9:0000%0]/api/moku/summary'
$: python my-script.py --my-moku='[fe80::7269:79ff:feb9:0000%0]'
```
</code-block>

<code-block title="CMD/PowerShell">
```bash
> curl https://[fe80::7269:79ff:feb9:0000%0]/api/moku/summary
> python.exe my-script.py --my-moku=[fe80::7269:79ff:feb9:0000%0]
```
</code-block>
</code-group>

### Configuring a Proxy

#### Windows

Windows comes with a `portproxy` that can be used to listen on an IPv4 address (that your restricted environment can connect to) and forward it to an IPv6 one (like the Moku over USB).

Open `Powershell` or `CMD.exe` _as Administrator_ and enter the below. Replace the `connectaddress` with the IPv6 address of your Moku's USB interface, e.g. by using the "Copy IP Address" option from the Desktop App. Remember to enclose the address in square brackets.

```powershell
PS > netsh interface portproxy set v4tov6 `
listenport=8090 `
connectaddress=[fe80:0000:0000:0000:7269:1234:5678:0000%0] `
connectport=http
```

You can then use the address `localhost:8090` to connect using the API.

If this command doesn't work, check that no other service is operating on port `8090`. You can change this number as required (typically a number between 1024 and 65535 on which nothing else is running), just make sure the `listenport` and the port specifier in the address match.

#### Mac

Coming Soon

#### Linux

Coming Soon
