# Moku:Delta Gigabit Streamer Demo - Setup Guide
*End-to-end configuration for real-time DIFI UDP capture and playback on commercially available Linux system*

## Overview
This guide documents the hardware, software, and OS configuration required to capture and replay Moku:Delta Gigabit Streamer.  The following procedures and configuration along with the code in this repo are validated to produce lossless collections of data over SFP+ (10 Gigabit Ethernet) at sample rates up to 312.5 MSa/s per channel.  This demonstration is meant to introduce the basic functionality and to quickly get the user up and running with gigabit streamer.  More detailed information is available in our [Gigabit Streamer App Note](https://liquidinstruments.com/application-notes/moku-gigabit-streamer-to-host-guide/).  

The recommended sequence for best utilizing this repo is as follows:

1. Setup hardware connection as listed in [Connecting the Hardware](#connecting-the-hardware) section
3. Establish static IP as described in [Transmission Configuration with Static IP Addressing](#transmission-configuration-with-static-ip-addressing) section
4. Install software utilities as described in [Moku Streamer Utility Installation](#moku-streamer-utility-installation) section

The toolchain for **Moku Streamer** consists of two programs:
- capture_it  — high-performance C++ UDP capture
- send_it     — real-time C++ DIFI packet transmitter with ±30 ns timing jitter

All tools are built from source in a single directory.

---

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

---

## Connecting the Hardware
 
Streaming uses a **direct point-to-point Ethernet** link — no switches or routers required.
 
**For Moku Gigabit Streamer (SFP):**
- A compatible 10G SFP+ or SFP28 DAC cable
- A host NIC supporting that DAC cable

**For Moku Gigabit Streamer+ (QSFP):**
- A compatible 100G QSFP28 DAC cable
- A host NIC supporting that DAC cable

When the link is established, **LED 3** on the Moku front panel turns **blue**.
 
Verify on the host (replace `enp5s0f0np0` with your interface name):
 
```bash
# 1. List interfaces and find the SFP/QSFP one
ip addr
# Look for: enp5s0f0np0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 ...
# UP and LOWER_UP indicate enabled with link
 
# 2. Confirm physical link and speed
sudo ethtool enp5s0f0np0
```
 
Key fields to check in `ethtool`:
 
- `Link detected: yes`
- `Speed: 10000Mb/s` (SFP) or `100000Mb/s` (QSFP)
- `Duplex: Full`

---
 
## QSFP Configuration for Moku Gigabit Streamer+
 
The QSFP interface on Moku:Delta does **not** use Forward Error Correction (FEC). If the host NIC enables FEC by default, the link may not come up.
 
Disable FEC on the host:
 
```bash
sudo ip link set <interface> down
sudo ethtool --set-fec <interface> encoding off
sudo ip link set <interface> up
```
 
After this the interface should report link-detected and the negotiated speed.
 
---
 
## Transmission Configuration with Static IP Addressing
 
The Gigabit Streamer uses **static IP addressing** on the streaming ports — no ARP or DHCP. The Moku and the host must share a subnet on the streaming interface.
 
- **Transmit mode (Moku → host):** the host listens for UDP packets on the configured port. Usually no extra host-side network configuration is required.
- **Receive mode (host → Moku):** the host must add a static neighbor (ARP) entry mapping the Moku's streaming IP to its MAC address. This must be redone each time the Receive-mode link is re-established (cable replug, host reboot, Moku reboot).
Check the neighbor table:
 
```bash
ip neighbor | grep <moku_NIC_IP>
```
 
If the entry is `INCOMPLETE`, replace it:
 
```bash
sudo ip neigh replace <moku_NIC_IP> lladdr <moku_MAC> dev <interface> nud permanent
```
 
**Example.** Configure the Gigabit Streamer in the Moku app with:
 
- Local IP: `10.10.1.1`
- Local MAC: `70:69:79:b2:02:69`
…and the host transmits via `enp5s0f0np0`:
 
```bash
sudo ip neigh replace 10.10.1.1 lladdr 70:69:79:b2:02:69 dev enp5s0f0np0 nud permanent
```

**Note.** The settings tab on the Moku Streamer GUI will allow you to easily configure and verify the static IP settings.

---

## Packet and data structure
This Readme will not go into detail on the packet and data structure used by Gigabit Streamer as it is covered in detail in the linked [Application Note](https://liquidinstruments.com/application-notes/moku-gigabit-streamer-to-host-guide/).  Gigabit Streamer uses a DIFI-aligned VITA-49.2 packet format for all transmitted and received sample data, transported using UDP over IPv4.  

---

## Transmitting Data from Moku to a Host
 
The Gigabit Streamer supports simultaneous TX and RX on the same interface; this section covers TX setup.
 
### Moku Configuration (Transmit)
 
Signal sources: routed via Multi-Instrument Mode, or for **Gigabit Streamer+ stand-alone** the analog inputs by default. For analog sources, confirm input range, coupling, and attenuation are appropriate.
 
Open the network config (globe icon) in the Gigabit Streamer interface.
 
**Local:**
- **IP Address** — static IP for the SFP/QSFP port; must be on the same subnet as the host.
- **UDP Port** — used for receive only.
- **MAC Address** — fixed by Moku per port.

**Remote Destination:**
- **IP Address** — host NIC's IP address.
- **UDP Port** — port the host software listens on.
- **MAC Address** — host NIC's MAC (must be set explicitly; the Gigabit Streamer does not perform ARP).

**Outgoing Packages:**
- **Network MTU** — choose the largest MTU supported end-to-end.
- **UDP Payload** — computed from MTU and sample size.
- **Samples Per Packet** — computed from MTU and sample size.
> **UDP port note.** Any unused port is valid. VITA-49 / VRT workflows conventionally use **4991**, but it isn't required.
 
---
 
## Receiving Data on a Moku from a Host Computer
 
Use cases: sensor emulation, closed-loop control testing, wideband waveform playback.
 
The Moku does not participate in ARP and does not announce itself on the link. The host must construct packets that match the Gigabit Streamer's expected format and deliver them directly to the streaming interface.
 
### Host Configuration

The following command line commands can be executed from the settings tab on the Moku Streamer application running on Linux.  however, they are included here for context as well.
 
#### Host interface
 
```bash
# Show interfaces, IPs, and MACs
ip addr
 
# Assign a static IP to the NIC connected to the Moku
sudo ip addr add <sub_net> dev <interface>
sudo ip link set dev <interface> up
```
 
#### MTU configuration
 
Match the Moku's MTU (maximum accepted is **1500 bytes**):
 
```bash
sudo ip link set dev <interface> mtu 1500
```
 
A mismatched MTU causes packet drops.
 
#### Static neighbor entry (ARP replacement)
 
```bash
ip neighbor | grep <moku_NIC_IP>
 
# If INCOMPLETE, replace with a static entry:
ip neighbor replace <moku_NIC_IP> lladdr <MOKU_MAC> dev <interface> nud permanent
```
 
### Moku Configuration (Receive)
 
Only the Local fields matter; Remote Destination can be left unconfigured.
 
**Local:**
- **IP Address** — static IP for the Moku SFP/QSFP port; same subnet as the host.
- **Multicast Address** — see below.
- **UDP Port** — the host must send packets to this port. VITA-49.2 convention is **4991**, but any port that matches the transmitted packet destination is valid.
- **MAC Address** — fixed by Moku per port.

**Remote Destination (optional in receive-only):** IP, UDP Port, MAC — only if you also need TX.
 
**Outgoing Packages:**
- **Network MTU** — up to 1500 bytes.
- **UDP Payload** / **Samples Per Packet** — computed from MTU and sample size.

### Sending Data from the Host
 
Stream order:
 
1. **Context packet** — defines stream config. Moku accepts **16-bit** sample data on host-to-Moku transmission.
2. **Data packets** — continuous sequence with the same stream identifier; send at the configured sample rate.
3. **Packet sizing** — construct data packets to match the configured MTU.
The Moku does **not** perform rate adaptation or resampling. Apply rate control / throttling on the host so the send rate matches the configured stream rate. Steady packet timing avoids overflow/underflow that would disturb downstream instruments.
 
**Output routing.** The output connections of the Gigabit Streamer (between the Interpolation block and the output ports) are wired automatically based on the **Vector Size** field of the received Context packet — not through the Moku App. Outputs are populated starting from Output A, then Output B, and so on.
 
**I/Q handling (DIFI compatibility).** When a complex DIFI stream is received, the I and Q components are interpreted as two real-valued channels: **I → Output A**, **Q → Output B**.
 
---

## Moku Streamer Utility Installation

This section will describe the process for installing the Moku Streamer application along with suggested optimization to yield maximum performance.  

### Software Prerequisites
Install the required packages with a single apt command:
 
```bash
sudo apt update
sudo apt install -y build-essential chrony iproute2 ethtool python3-numpy cpufrequtils
```
 
Verify that g++ supports C++17:
 
```bash
g++ --version   # should be 11 or newer
```
 
---

### Kernel UDP Socket Buffer Limits

The default OS receive buffer cap is too small for high-rate capture. Raise it to 512 MiB and make the change persistent across reboots:
 
```bash
sudo sysctl -w net.core.rmem_max=536870912
sudo sysctl -w net.core.rmem_default=536870912
 
# Persist across reboots:
echo 'net.core.rmem_max=536870912'    | sudo tee -a /etc/sysctl.conf
echo 'net.core.rmem_default=536870912' | sudo tee -a /etc/sysctl.conf
```

### TAI Clock Offset (chrony)

DIFI timestamps use the International Atomic Time (TAI) scale. Linux must be told the current TAI–UTC offset (37 seconds as of this writing). Edit `/etc/chrony.conf` and add the following line, then restart chrony:
 
```
leapsectz right/UTC
```
 
```bash
sudo systemctl restart chrony
sudo systemctl enable chrony
```
 
Verify the offset is 37 seconds:
 
```bash
python3 -c "import time; ts=time.clock_gettime(time.CLOCK_TAI); \
ut=time.clock_gettime(time.CLOCK_REALTIME); print(f'TAI-UTC offset: {ts-ut:.0f}s')"
```

### NIC Tuning, IRQ Affinity, and CPU Governor

Create `/usr/local/bin/moku-setup.sh` with the following content. This script pins the NIC interrupt to a dedicated CPU core, disables adaptive coalescing for low-latency receive, and switches all cores to the performance governor.
 
```bash
#!/bin/bash
set -e
NIC=enp5s0f0np0
 
# ── disable adaptive interrupt coalescing ──────────────────────────────────
ethtool -C $NIC adaptive-rx off adaptive-tx off rx-usecs 0 tx-usecs 0
 
# ── pin NIC IRQ to CPU 3 (isolate from OS noise on CPUs 0-2) ───────────────
IRQ=$(grep $NIC /proc/interrupts | head -1 | awk -F: '{print $1}' | tr -d ' ')
if [ -n "$IRQ" ]; then
    echo 8 > /proc/irq/$IRQ/smp_affinity   # CPU 3 = bitmask 0x8
    echo "Pinned IRQ $IRQ to CPU 3"
fi
 
# ── performance CPU governor ────────────────────────────────────────────────
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo performance > $cpu
done
 
echo "moku-setup complete."
```
 
```bash
sudo chmod +x /usr/local/bin/moku-setup.sh
```
 
To run the script automatically at boot, create a systemd service:
 
```bash
sudo tee /etc/systemd/system/moku-setup.service <<'EOF'
[Unit]
Description=Moku NIC + CPU tuning
After=network.target
 
[Service]
Type=oneshot
ExecStart=/usr/local/bin/moku-setup.sh
RemainAfterExit=yes
 
[Install]
WantedBy=multi-user.target
EOF
 
sudo systemctl daemon-reload
sudo systemctl enable --now moku-setup.service
```
 
---