---
---
# Getting Started with MATLAB

The Moku Scripting API for MATLAB is available through the MATLAB File Exchange and the MATLAB Add-on Manager.

## 1. Install Toolbox

### Open the Add-on Manager
The Add-on Manager can be found on the Home > Environment tab.

![Add-on Manager](./img/starting-matlab-1.png)

### Search for `Moku`
The toolbox name is `Moku-MATLAB` and will generally be the only result.

![Search Results](./img/starting-matlab-2.png)

### Install Toolbox
Click on the search result and select 'Add' on the right-hand side. When installation has completed, the button will change to 'Manage' and the green 'Installed' Badge will be added to the icon on the left.

![Installed](./img/starting-matlab-3.png)

### Check the Search Paths
The Moku Scripting API for MATLAB requires that it be able to access some packaged data files. Depending on your MATLAB configuration, the Toolbox path may not have been added to your file search path to facilitate this.

Select `Set Path` from the Home > Environment tab (next to the Add-On Manager).

![Search Paths](./img/starting-matlab-4.png)

Ensure that there is an entry pointing to the toolbox installation location. A typical path might be `C:\Users\<username>\AppData\Roaming\Mathworks\MATLAB Add-Ons\Toolboxes\Moku-MATLAB` as shown in the image below.

![Search Paths](./img/starting-matlab-4a.png)

## 2. Download the data files
The Moku Scripting API for MATLAB requires data files to be downloaded before any program can be run. These data files may be several hundred megabytes. Please ensure you have a suitable internet connection before you proceed, 
this step is only required to be run whenever you first install or upgrade the library.

From your MATLAB Command Window, issue the download command. This may take a while to complete, depending on your internet connection. 
```
$: moku_download(###)
```

The ### should be replaced with the current firmware version. You can find the current firmware version through the Moku: desktop app by right clicking on your Moku and hovering the mouse over 'Device info'.

## 3. Test Installation

From your MATLAB Command Window, run `help Moku`. If this command succeeds, then the toolbox has been successfully installed, otherwise refer to the Troubleshooting instructions below.

![help Moku](./img/starting-matlab-5.png)

## 4. Find Your IP Address
In order to connect to your Moku, you must know your device's IP Address. For full details and options, see [Finding your IP Address](ip-address.html).


## 5. Next Steps
Visit our [MATLAB Examples](/examples/matlab/) for ready-to-run scripts to get started with instruments.

For a full listing of all objects and methods, with example snippets for both Python and MATLAB, see our [API Reference](/reference/).

## Troubleshooting
#### IPv6 (including USB) Connection Issues
There are some environmental limitations when using IPv6, including using the Moku USB interface. See [this section](/ip-address.html#ipv6) for more information.
