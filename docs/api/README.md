---
title: 'Moku API Home'
---

# Moku Scripting API

The Moku device family from Liquid Instruments is the next generation of Test. With the Moku Scripting API, command and control of your test has never been easier.

To start, install the API for [Python](getting-started/starting-python.md), [MATLAB](getting-started/starting-matlab.md) or [LabVIEW](getting-started/starting-labview.md).

## Get Started

### Python

The Moku Scripting API for Python requires Python 3.5 or newer.

:::warning
Starting with firmware version 600, there are some breaking changes that require you to update your Python client package to version **3.3.1**. [Read more](getting-started/starting-python.md#important-notice-breaking-changes-in-firmware-600)
:::
<action-button text="Start with Python" link="/api/getting-started/starting-python"/>

### MATLAB

The Moku Scripting API for MATLAB requires MATLAB 2014b or newer.
<action-button text="Start with MATLAB" link="/api/getting-started/starting-matlab"/>

## Features

### RESTful API

The Moku Scripting API is built around a RESTful HTTP interface, allowing for access from a wide range of programming languages. It makes remote access easy, as the standard HTTP protocol is low bandwidth and commonly allowed to traverse firewalls.

Support for the next generation of Moku products. The Scripting API supports almost all the same features of the Moku iPad and Desktop Applications; if you need something that's missing, visit the [Knowledge Base](https://knowledge.liquidinstruments.com/) for further information and support.

Python and MATLAB libraries are provided that wrap the RESTful API in a way that feels natural to programmers of all levels.

### Device Discovery

The Python Scripting API comes bundled with the `mokucli` command line tool for device discovery. Quickly find and connect to the right device without worrying about network configurations. Install from our [Utilities](https://www.liquidinstruments.com/software/utilities/) page and read more about [mokucli](../cli/).

```text
$ mokucli --help

 Usage: mokucli [OPTIONS] COMMAND [ARGS]...

 Moku command line utility

 Version: 4.0.1

 (c) Liquid Instruments 2016-2025

╭─ Options ─────────────────────────────────────────────────────────────────╮
│ --version                                                                 │
│ --install-completion          Install completion for the current shell.   │
│ --show-completion             Show completion for the current shell, to   │
│                               copy it or customize the installation.      │
│ --help                        Show this message and exit.                 │
╰───────────────────────────────────────────────────────────────────────────╯
╭─ Commands ────────────────────────────────────────────────────────────────╮
│ instrument   Manage Moku instrument bitstreams                            │
│ firmware     Fetch and upload Moku firmware                               │
│ feature      Manage Moku features                                         │
│ config       Configuration file management commands                       │
│ command      Execute a single Moku API command.                           │
│ convert      Convert Liquid Instruments binary data files to standard     │
│              formats.                                                     │
│ download     Download bitstreams (backward compatibility).                │
│ list         Search for Moku devices on the local network.                │
│ files        Manage files on Moku                                         │
│ license      Manage Moku licenses                                         │
│ proxy        Create a network proxy from a local IPv4 address to a Moku   │
│              device                                                       │
│ stream       Stream real-time data from a Moku device. This is usually    │
│              used through a language binding, e.g. the Python API.        │
╰───────────────────────────────────────────────────────────────────────────╯
```

## Known Issues

<!-- ### Firmware Updates

The Scripting API is not currently able to update the firmware on Moku devices. Please use either the Moku Desktop software or iPad App. Updating firmware through the applications may also require that you update your Scripting API libraries. -->

### File Conversion

The LI File Converter application is available with installation of the Moku Desktop software or as a standalone application. Conversion cannot currently be done through the Scripting API.

### Other Languages

The Moku Scripting API is built around an HTTP/REST interface. This allows easy interfacing from any programming language, however full documentation of this REST interface is still underway. If you have specific needs, again, reach out to our [Support Engineers](mailto:support@liquidinstruments.com).

### USB and IPv6 Supported Environments

You may wish to use IPv6 on your network, and it must be noted that the USB Connection uses IPv6 internally. IPv6 is not universally supported, and as such the API cannot be used over USB in some environments. A non-exhaustive list is

-   Windows Subsystem for Linux Version 2 (WSL2), as discussed with workarounds [here](https://github.com/microsoft/WSL/discussions/5855)
-   LabVIEW, as discussed [here](https://forums.ni.com/t5/LabVIEW-Idea-Exchange/Native-support-for-IPv6/idi-p/1481942)

If you require API connectivity from these environments, you must use a network connection like Ethernet or WiFi, including a point-to-point network with Static IPs if security is a concern.

## Legacy Moku:Lab

If you are using a firmware version <=511, consider upgrading to recieve the most up to date features and fixes.

For Moku:Lab versions **<= 511**, please refer to our legacy APIs

-   For Python, [pymoku](https://pypi.org/project/pymoku/)
-   For MATLAB, [MATLAB Legacy](https://www.liquidinstruments.com/resources/software-utilities/matlab-api/)
