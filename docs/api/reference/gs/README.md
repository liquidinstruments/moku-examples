---
title: Gigabit Streamer / + Instrument
prev: false
available_on: 'Moku:Delta'
---


# Gigabit Streamer (Gigabit Streamer+)

The Gigabit Streamer instrument streams data by transmitting and/or receiving UDP packets through the SFP and QSFP ports at gigabit speeds. There are two versions of Moku Gigabit Streamer:
- **Gigabit Streamer** streams through up to two 10 Gbit/s SFP ports
- **Gigabit Streamer+** streams through the QSFP port at higher gigabit speeds (subject to export control) 

If you are using the Moku Gigabit Streamer instrument directly through the REST API, the instrument name as used in the URL is `gs`(and `gsp` for the Gigabit Streamer + variant).

::: tip Gigabit Streamer Plus
If you are using the Moku Gigabit Streamer+ instrument, all commands are the same as for the Gigabit Streamer Instrument. Call your Gigabit Streamer+ instrument with the following object names:
- **Python:** GigabitStreamerPlus 
- **Matlab:** MokuGigabitStreamerPlus
- **cURL:** gsp
:::

**Note on enabling outputs**
The latch viewable in the Moku application is forcibly enabled whenever you call set_output(). Manually disable/enable each output in sequence to effectively gate off/on all outputs.

<function-index/>


