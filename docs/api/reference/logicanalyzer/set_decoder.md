---
additional_doc: null
description: Configure decoder on the given channel
deprecated: true
deprecated_msg: set_decoder is deprecated, use [set_uart_decoder](./set_uart_decoder.md), [set_spi_decoder](./set_spi_decoder.md) or [set_i2c_decoder](./set_i2c_decoder.md) instead.
method: post
name: set_decoder
parameters:
    - default: null
      description: Id of channel to configure decoder on
      name: channel
      param_range: 1 to 2
      type: integer
      unit: null
    - default: null
      description: Type of protocol to configure
      name: protocol
      param_range: UART, SPI, I2C
      type: string
      unit: null

    - default: null
      description: Pin number to configure as a data pin
      name: data_pin
      param_range: 1 to 16
      type: number
      unit: null
    - default: undefined
      description: Bit order for UART/SPI. When undefined, sets LSB for UART and MSB for SPI
      name: lsb_first
      param_range: null
      type: boolean
      unit: null
    - default: 8
      description: Number of data bits. Cannot be more than 8 if parity bit is enabled
      name: data_width
      param_range: 5 to 9
      type: number
      unit: null
    - default: 1
      description: Number of stop bits.
      name: uart_stop_width
      param_range: 1 to 2
      type: number
      unit: null
    - default: None
      description: Parity for UART.
      name: uart_parity
      param_range: None, Even, Odd
      type: string
      unit: null
    - default: 9600
      description: UART baud rate.
      name: uart_baud_rate
      param_range: 1 to 2e6
      type: number
      unit: baud
    - default: undefined
      description: Pin number to send clock signal. Only for SPI/I2C
      name: clock_pin
      param_range: 1 to 16
      type: number
      unit: null
    - default: undefined
      description: Chip select for SPI protocol
      name: spi_cs
      param_range: 1 to 16
      type: number
      unit: null
    - default: 0
      description: Clock Polarity (1 for High and 0 for Low) for SPI protocol
      name: spi_cpol
      param_range: 0 to 1
      type: number
      unit: null
    - default: 0
      description: Clock Phase (1 for trailing edge and 0 for leading edge) for SPI protocol
      name: spi_cpha
      param_range: 0 to 1
      type: number
      unit: null
    - default: true
      description: Disable all implicit conversions and coercions.
      name: strict
      param_range: null
      type: boolean
      unit: null
summary: set_decoder
---

<headers/>
<parameters/>

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import LogicAnalyzer
i = LogicAnalyzer('192.168.###.###')
patterns = [{"pin": 2, "pattern": [0, 1, 0, 1]}]
# Configure PG1 to generate pattern on pin2
i.set_pattern_generator(1, patterns=patterns, divider=12)
i.set_pin(2, "PG1")
i.set_decoder(1, "UART", 2)
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuLogicAnalyzer('192.168.###.###');
patterns=[struct('pin',1,'pattern',[0,1,0,1])];
% Configure PG1 to generate pattern on pin2
m.set_pattern_generator(1, 'patterns', patterns, 'divider', 12);
m.set_pin(2, "PG1");
m.set_decoder(1, "UART", 2);
```

</code-block>

<code-block title="cURL">

```bash
# If the pattern is longer, consider putting the data in a JSON file
# rather than passing on the command line
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"channel":1,"protocol":"UART","data_pin":2}'\
        http://<ip>/api/logicanalyzer/set_decoder
```

</code-block>

</code-group>

### Sample response

```json
{
    "data_pin": "pin2",
    "data_width": 8,
    "lsb_first": true,
    "protocol": "UART",
    "uart_baud_rate": 9600,
    "uart_parity": "None",
    "uart_stop_width": 1
}
```
