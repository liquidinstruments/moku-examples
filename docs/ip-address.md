# Finding Your IP Address
There are many ways to find your Moku's IP Address. You may use the Desktop App (if installed) or the Moku CLI tool (bundled with the Moku Scripting API for Python). Additionally, if you're using the USB Connection to your Moku, the IP Address can be calculated based on the serial number.

## Using the Desktop App
- Open the Desktop Application
- On the "Select Device" screen, find the tile of the Moku to which you wish to connect
- Right-click and select `Device Info > Copy IP Address`

## Moku CLI
:::tip Python Only
The `moku` Command Line Utility is installed as part of the Moku Scripting API for Python. If you're using MATLAB then you may choose to install the Python library additionally, or use one of the other methods listed on this page
:::

At a terminal or command prompt, use the Moku CLI tool to list all the Moku devices your computer can see.

```
$: moku list
```

## USB Connection
:::warning Moku:Go Only
The Moku:Pro doesn't yet support API over USB. Please connect using Ethernet or WiFi, and use one of the above methods to discover the IP address
:::

If your Moku is connected over USB, it will have a pre-determined IPv6 address based on its serial number. You can find this address by entering your serial number in to the calculator below.

The Serial Number on a Moku:Go is found in three places:
- The six digits after the underscore on the top line of the barcode sticker underneath your Moku:Go
- The digits after the hyphen in the device's WiFi Access Point SSID
- In the `Device Info` drop-down when you right-click a Device Tile in "Select Device" screen of the Moku Desktop Application

<ip-calculator/>
