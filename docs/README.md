---
title: "Moku API Home"
---

# Moku Scripting API

The Moku device family from Liquid Instruments is the next generation of Test. With the Moku Scripting API, command and control of your test has never been easier.

To start, install the API for [Python](starting-python) or [MATLAB](starting-matlab).

:::warning Moku:Lab Compatibility
The Moku Scripting API is compatible with Moku:Pro and Moku:Go only.

For Moku:Lab support, please refer to our legacy APIs
- For Python, [pymoku](https://pypi.org/project/pymoku/)
- For MATLAB, [MATLAB Legacy](https://www.liquidinstruments.com/resources/software-utilities/matlab-api/)
:::

## Get Started
### Python
The Moku Scripting API for Python requires Python 3.5 or newer.
<action-button text="Start with Python" link="starting-python"/>

### MATLAB
The Moku Scripting API for MATLAB requires MATLAB 2014b or newer.
<action-button text="Start with MATLAB" link="starting-matlab"/>

## Features
### RESTful API
The Moku Scripting API is built around a RESTful HTTP interface, allowing for access from a wide range of programming languages. It makes remote access easy, as the standard HTTP protocol is low bandwidth and commonly allowed to traverse firewalls.

Python and MATLAB libraries are provided that wrap the RESTful API in a way that feels natural to programmers of all levels.

### Moku:Pro and Moku:Go
Support for the next generation of Moku products, with Moku:Lab support coming soon. The Scripting API supports almost all the same features of the Moku iPad and Desktop Applications; if you need something that's missing, please reach out to our [Support Engineers](mailto:support@liquidinstruments.com).
### Device Discovery
The Python Scripting API comes bundled with the `moku` command line tool for device discovery. Quickly find and connect to the right device without worrying about network configurations.

## Known Issues
### Firmware Updates
The Scripting API is not currently able to update the firmware on Moku devices. Please use either the Moku Desktop software or iPad App. Updating firmware through the applications may also require that you update your Scripting API libraries.

### File Conversion
The LI File Converter application is available with installation of the Moku Desktop software or as a standalone application. Conversion cannot currently be done through the Scripting API.

### Moku:Pro Instrument Support
Some instruments outside the Moku:Pro Base Bundle are not currently supported by the Scripting API. We're currently working to address this, however if you have a specific need, please reach out to our [Support Engineers](mailto:support@liquidinstruments.com).

### Other Languages
The Moku Scripting API is built around an HTTP/REST interface. This allows easy interfacing from any programming language, however full documentation of this REST interface is still underway. If you have specific needs, again, reach out to our [Support Engineers](mailto:support@liquidinstruments.com).
