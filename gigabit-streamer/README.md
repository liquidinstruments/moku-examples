# Moku:Delta Gigabit Streamer Demo - Setup Guide
*End-to-end configuration for real-time DIFI UDP capture and playback on commercially available Linux system*

## Overview
This guide documents the hardware, software, and OS configuration required to capture and replay Moku:Delta Gigabit Streamer data over SFP+ (10 Gigabit Ethernet) at sample rates up to 312.5 MSa/s per channel.  This demonstration is meant to introduce the basic functionality and to quickly get the user up and running with gigabit streamer.  More detailed information is available in our [Gigabit Streamer App Note](https://liquidinstruments.com/application-notes/moku-gigabit-streamer-to-host-guide/).  

The toolchain consists of two programs:
- capture_it  — high-performance C++ UDP capture (replaces captureDataBuffer.py)
- send_it     — real-time C++ DIFI packet transmitter with ±30 ns timing jitter

Both tools are built from source in a single directory and require no external libraries beyond the Linux system headers.

## Example Hardware Requirements
The setup was developed and tested on the following platform. Other x86-64 systems with a supported 10 GbE NIC will work, but IRQ affinity pin numbers and NIC interface names may differ.

| Component | Description |
|-----------|-------------|
|System | Minisforum MS-A2 mini-PC|
|CPU | AMD Ryzen 9 9955HX(16 cores, up to 5.4 GHz)|
|NIC | Intel X710-DA2 10GbE SFP+ (i40e driver)|
|Storage | 2TB NVMe SSD (> 3GB/s sustained write recommended for max rate)|
|OS/Kernel | Ubuntu 24.04 LTS, Kernel 6.17+|

<p align="center"> <b>Note:</b> A dedicated NIC is strongly recommended.  The interface should carry only Moku traffic; other interfaces handle management and internet access. </p>

### CPU and interrupt handling
High-rate UDP capture relies on the host CPU to receive packets, buffer incoming data, and pass samples into user space. For reliable streaming, the host system should provide sufficient processing capacity to keep up with the incoming data rate.

In practice, this is best achieved with:

- A multi-core CPU with good single-thread performance
- Network interface card drivers that support receive-side scaling
- Well-distributed interrupts to ensure processing load is shared across CPU cores

For higher-rate QSFP links, workstation- or server-class processors will likely be required to offer additional headroom, particularly for long-duration or continuous captures.

### System memory
Buffering incoming network data relies on available system memory to absorb short packet bursts and accommodate momentary delays when writing data to disk. Providing adequate memory helps maintain continuous, loss-free streaming.

### Storage throughput
Reliable storage throughput helps maintain smooth data flow throughout the capture process, the appropriate storage configuration depends on the target streaming rate and the intended capture duration. For most high-speed streaming applications, this is achieved by using storage solutions that support continuous, large-block sequential writes at the required throughput. At higher line rates, additional storage bandwidth may be beneficial, such as higher-performance devices or multiple drives operating together.

