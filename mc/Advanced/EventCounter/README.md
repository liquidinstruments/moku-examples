# Event Counter

Counts the number of pulses of defined width in a period. If a count threshold is exceeded, an output flag is raised.

The parameters of the pulses (minimum and maximum width, minimum height) are configurable from Control Registers, as is the overall measurement period.

## Pintout and Registers

### Pinout

| Pin | Use |
| --- | --- |
| Input A | Pulse train input |
| Input B | Not used |
| Output A | Count threshold exceeded |
| Output B | Count threshold *not* exceeded (`not OutputA`) |

The values specific output when the count is and is not exceeded are defined as constants in the code.

## Registers
![Timing Diagram](./waveform.png)

|              |                     **Bits**                    ||
| **Register** | `31-16`                 | `15-0`                |
| ------------ | :---------------------: | :-------------------: |
| Control0     |                `t1:` clock cycles               ||
| Control1     | `tpmax` clock cycles    | `tpmin` clock cycles  |
| Control2     | `mincount` count        | `vpmin` ADC Bits      |

