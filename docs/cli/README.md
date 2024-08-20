---
tags: ['CLI', 'Moku CLI', 'utilities', 'command line utility']
title: 'Moku CLI'
---

# Moku CLI

Moku CLI (`mokucli`) is a command line utility to quickly access various features of the Moku hardware. It is also used internally by various client driver packages for functions such as decoding data streams.

The latest packages are available to download from [Utilities](https://www.liquidinstruments.com/software/utilities/). The installation wizard will configure everything needed to launch the CLI successfully.

:::warning
Moku CLI commands are evolving and subject to change. It is not recommended to script your interactions with this utility.
:::

## Search path

When using the installers above for Windows or Mac, the `mokucli` will be discoverable from any client driver package, e.g. MATLAB or Python.

For **Linux**, the user must manually either

1. Create a symbolic link to `mokucli` in the `/usr/local/bin` directory, or
2. Set `MOKU_CLI_PATH` environment variable to the absolute path of `mokucli`.
