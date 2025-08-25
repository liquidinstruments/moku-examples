---
title: Getting Started with Multi-instrument Mode
---

# Starting with Multi-instrument Mode

Multi-instrument Mode (MiM) allows a user to use more than one instrument simultaneously on the Moku platform. Depending on the hardware and instrument combination, different instrument counts and capabilities are available in this mode. For example, Moku:Go supports two instruments simultaneously while Moku:Pro supports four; the Moku:Pro Lock-in Amplifier loses the PID Controller Moku:Go's also loses the Phase Locked Loop external reference option.

There are several steps required in order to use MiM:

1. Select the Multi-instrument Mode configuration you wish to use
2. Load instruments in to one or more of the slots
3. Configure connections from each instrument to other instruments and/or external I/O
4. Configure the external I/O (i.e. ADCs, DACs and/or Digital I/O) with the correct settings for gain, direction, impedance etc.

Depending on which programming language you choose, each step may look a little different.

:::tip MATLAB, LabVIEW Support
Multi-instrument Mode is available through all of our APIs, including MATLAB and LabVIEW. This guide currently covers Python and cURL usage, with more language examples coming soon.
:::

## Selecting the Multi-instrument Mode Configuration

The configuration is parameterized by the number of slots you wish to have. Specifying "1" is equivalent to disabling Multi-instrument Mode.

|  Device    | Slots Available |
| :--------: | :-------------: |
| Moku:Go    |      1, 2       |
| Moku:Lab   |      1, 2       |
| Moku:Pro   |      1, 4       |
| Moku:Delta |      1, 3, 8    |

If using the REST API directly (i.e. not the Python, LabVIEW or MATLAB packages), the user is responsible for obtaining a Client Key before entering Multi-instrument Mode. See the [REST API Getting Started Guide](./starting-curl.md) for more information.

<code-group>

<code-block title="Python">

```python
# In Python, you specify the configuration by creating a MultiInstrument object
from moku.instruments import MultiInstrument
mim = MultiInstrument('192.168.###.###', force_connect=True, platform_id=2)
```

</code-block>

<code-block title="cURL">

```bash
# The platform ID is specified as a URL component
curl -H 'Moku-Client-Key: <key>' http://<ip>/api/moku/platform/2
```

</code-block>

</code-group>

## Loading Instruments to Slots

If using our packaged drivers, like for Python, then the loading and addressing of instruments in specific slots is taken care of by the library, simply request an instrument object from the `MultiInstrument` class as below rather than creating one directly.

When interacting with an instrument directly from the REST API, the path to the instrument is prefixed with the slot in to which it has been put. For example, if you have Waveform Generators in both slots `1` and `2`, then generating waveforms from each of the two instruments would use the paths `/api/slot1/waveformgenerator/generate_waveform` and `/api/slot2/waveformgenerator/generate_waveform`. To trigger the load of an instrument to a slot, you just need to `GET` the instrument's base path, e.g. a `GET` to `http://<ip>/api/slot1/waveformgenerator` will trigger the load of a Waveform Generator instrument in to Slot 1.

<code-group>

<code-block title="Python">

```python
wg = mim.set_instrument(1, WaveformGenerator)
osc = mim.set_instrument(2, Oscilloscope)
```

</code-block>

<code-block title="cURL">

```bash
curl -H 'Moku-Client-Key: <key>' http://<ip>/api/slot1/waveformgenerator
curl -H 'Moku-Client-Key: <key>' http://<ip>/api/slot2/oscilloscope
```

</code-block>

</code-group>

## Configuring Connections

Connections are specified as a list of maps, where each map specifies a `source` and `destination` pair. The following examples set up a routing from Input1 (ADC channel 1) to the first input of Slot 1, the first output of Slot 1 to the first input of Slot 2, then the first output of Slot 2 to Output1 (DAC channel 1).

See also the documentation for [set_connections](../reference/mim/set_connections.md).

<code-group>

<code-block title="Python">

```python
wg = mim.set_connections([
    {"source": "Input1", "destination": "Slot1InA"},
    {"source": "Slot1OutA", "destination": "Slot2InA"},
    {"source": "Slot2OutA", "destination": "Output1"},
])
```

</code-block>

<code-block title="cURL">

```bash
# It's recommended to put the JSON describing your connection structure in to a
# file rather than specifying it directly on the command line, it can get very long
# : echo connections.json
[
    {source: "Input1", destination: "Slot1InA"},
    {source: "Slot1OutA", destination: "Slot2InA"},
    {source: "Slot2OutA", destination: "Output1"},
]
curl -H 'Moku-Client-Key: <key>' \
  -H 'Content-Type: application/json' \
  --data @connections.json \
  http://<ip>/api/mim/set_connections
```

</code-block>

</code-group>

## Configuring External I/O

External I/O such as ADCs, DACs and Digital I/O cannot be configured by the instruments themselves when in Multi-instrument Mode. For example, the Oscilloscope cannot set the input range, as that range might be shared by multiple instruments. As such, settings front-end (ADC) settings, output (DAC) settings, and Digital I/O directions, are all done on the Multi-instrument level. If you attempt to do these operations on an individual instrument while in Multi-instrument Mode, an error will be raised.

See also the documentation for the Multi-instrument Mode versions of [set_frontend](../reference/mim/set_frontend.md), [set_output](../reference/mim/set_output.md) and [set_dio](../reference/mim/set_dio.md).

<code-group>

<code-block title="Python">

```python
mim.set_output(1, '14dB')
mim.set_frontend(1, impedance="50Ohm", attenuation="0dB", coupling="DC")
```

</code-block>

<code-block title="cURL">

```bash
curl -H 'Moku-Client-Key: <key>' http://<ip>/api/mim/set_output -d '{channel: 1, gain: "14dB"}'
curl -H 'Moku-Client-Key: <key>' http://<ip>/api/mim/set_frontend -d '{channel: 1, impedance: "50Ohm", attenuation: "0dB", coupling: "DC"}'
```

</code-block>

</code-group>

## Next Steps

The instruments can now be controlled in the same way as usual. If using a packaged driver like Python, just ensure that you use the instrument object that is returned from the MiM connection, rather than creating a new one. If using the REST API directly, make sure to specify the slot in the URL.

Connection, front-end, output and DIO settings can all be changed dynamically without interrupting the running instruments. New instruments can be loaded in to slots using the procedure above without interrupting the instruments in other slots.
