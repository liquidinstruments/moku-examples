# Getting Started with MATLAB

The Moku Scripting API for MATLAB is available through the MATLAB File Exchange and the MATLAB Add-on Manager.

## 1. Install Toolbox

### Open the Add-on Manager

The Add-on Manager can be found on the Home > Environment tab.

![Add-on Manager](../../img/starting-matlab-1.png)

### Search for `Moku`

The toolbox name is `Moku-MATLAB` and will generally be the only result.

![Search Results](../../img/starting-matlab-2.png)

### Install Toolbox

Click on the search result and select 'Add' on the right-hand side. When installation has completed, the button will change to 'Manage' and the green 'Installed' Badge will be added to the icon on the left.

![Installed](../../img/starting-matlab-3.png)

### Check the Search Paths

The Moku Scripting API for MATLAB requires that it be able to access some packaged data files. Depending on your MATLAB configuration, the Toolbox path may not have been added to your file search path to facilitate this.

Select `Set Path` from the Home > Environment tab (next to the Add-On Manager).

![Search Paths](../../img/starting-matlab-4.png)

Ensure that there is an entry pointing to the toolbox installation location. A typical path might be `C:\Users\<username>\AppData\Roaming\Mathworks\MATLAB Add-Ons\Toolboxes\Moku-MATLAB` as shown in the image below.

![Search Paths](../../img/starting-matlab-4a.png)

## 2. Download the data files

### Install the `mokucli` Utility

Install `mokucli` by downloading the installer from [Utilities](https://www.liquidinstruments.com/software/utilities/). You can confirm that the installation succeeded by running the command listed below from your command line. If the output begins the same as the output shown below, then the installation has succeeded. Read more about [Moku CLI (mokucli)](../../cli/) command line features.

```bash
$ mokucli --help

 Usage: mokucli [OPTIONS] COMMAND [ARGS]...

 Moku command line utility

 Version: 4.0.1

 (c) Liquid Instruments 2016-2025
...
```

If the output does not match above, please refer to our [Knowledge Base](https://knowledge.liquidinstruments.com/installation-and-troubleshooting-of-mokucli) for troubleshooting.

::: tip Note
It's important to make sure that the mokucli installation is in the same environment as, or is accessible by the moku package. Please refer to [Installation and troubleshooting of mokucli](https://knowledge.liquidinstruments.com/installation-and-troubleshooting-of-mokucli) for more information.
:::

### Download the data files

The Moku Scripting API for MATLAB requires data files to be downloaded before any program can be run. These data files may be several hundred megabytes. Please ensure you have a suitable internet connection before you proceed, this step is only required to be run whenever you install or upgrade the library.

You can download the files using `mokucli instrument download`, read more about [mokucli](../../cli/instrument.md#mokucli-instrument-download).

```bash
# download instrument bitstreams for MokuOS 4.0.3
$: mokucli instrument download 4.0.3
ℹ Resolved Version 4.0.3 to instruments build 18673
  Downloading 362 bitstream(s) matching 'all'... ━━━━━━━━━━━━━━━━━ 100% 0:00:00
✓ Downloaded 362/362 bitstream(s)
```

The `4.0.3` should be replaced with the current Moku OS version. You can find the current Moku OS version through the Moku: desktop app by right clicking on your Moku and hovering the mouse over 'Device info'.

## 3. Test Installation

From your MATLAB Command Window, run `help Moku`. If this command succeeds, then the toolbox has been successfully installed, otherwise refer to the Troubleshooting instructions below.

![help Moku](../../img/starting-matlab-5.png)

## 4. Find Your IP Address

In order to connect to your Moku, you must know your device's IP Address. For full details and options, see [Finding your IP Address](./ip-address.md).

## 5. Next Steps

Visit our [MATLAB Examples](../moku-examples/matlab-api/) for ready-to-run scripts to get started with instruments.

For a full listing of all objects and methods, with example snippets for both Python and MATLAB, see our [API Reference](../reference/).

## Troubleshooting

#### IPv6 (including USB) Connection Issues

There are some environmental limitations when using IPv6, including using the Moku USB interface. See [this section](./ip-address.md#ipv6) for more information.
