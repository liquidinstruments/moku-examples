# Input and Output

The Moku Compile design, as defined in the [entity wrapper](./wrapper.html#custominstrument-architecture), has four 16-bit inputs and four 16-bit outputs (`InputA-D` and `OutputA-D` respectively). These signals can be routed from/to physical ADC/DAC channels on the hardware, Digital I/O where supported, or from other instruments in the Multi-Instrument Mode configuration.

## Clk and Reset

|       | Moku:Delta  | Moku:Pro    | Moku:Lab    | Moku:Go     |
| ----- | ----------- | ----------- | ----------- | ----------- |
| Clk   | 312.5MHz    | 312.5MHz    | 125MHz      | 31.25MHz    |
| Reset | Active-High | Active-High | Active-High | Active-High |

## Inputs and Outputs

Input and Output ports on the wrapper can either be:

-   Connected to other ports through the Moku Application's block diagram
-   Permanently connected to particular ADCs or DACs, or
-   Not connected

|         | Moku:Delta (3- and 8-slots)   | Moku:Pro (4-slots)     | Moku:Lab (2- and 3-slots)     | Moku:Go (2-Slots) | Moku:Go (3-Slots) |
| ------- | ------------- | ------------- | ------------- | ----------------- | ----------------- |
| InputA  | Block Diagram | Block Diagram | Block Diagram | Block Diagram     | Block Diagram     |
| InputB  | Block Diagram | Block Diagram | Block Diagram | Block Diagram     | Block Diagram     |
| InputC  | ADC In 1      | Block Diagram | ADC In 1      | ADC In 1          | ADC In 1          |
| InputD  | ADC In 2      | Block Diagram | ADC In 2      | ADC In 2          | ADC In 2          |
| OutputA | Block Diagram | Block Diagram | Block Diagram | Block Diagram     | Block Diagram     |
| OutputB | Block Diagram | Block Diagram | Block Diagram | Block Diagram     | Block Diagram     |
| OutputC | Not Connected | Block Diagram | Not Connected | Block Diagram     | Not Connected     |
| OutputD | Not Connected | Block Diagram | Not Connected | Not Connected     | Not Connected     |

All Inputs and Outputs are 16-bit signed values. When the port is externally connected to Digital I/O, the signed 16-bit values should be interpreted simply as a 16-bit standard logic vector.

## Analog I/O Scaling

The digital resolution refers to the voltage interpretation of digital quantities, scaled from LSBs (least significant bits) to volts, for ADCs, DACs, and other instruments. For example, with a 16-bit value representing a 10 Vpp voltage range, the digital resolution is 2^16/10 = 6553.6 LSBs/volt. The following table assumes no ADC attenuation and no DAC gain are configured.

| Source/Sink      | Moku:Delta LSBs/volt | Moku:Pro LSBs/volt | Moku:Lab LSBs/volt | Moku:Go LSBs/volt |
| ---------------- | -------------------- | ------------------ | ------------------ | ----------------- |
| ADC              | 36440.0              | 29925.0            | 30000.0            | 6550.4            |
| DAC (50R)        | 36440.0              | 29925.0            | 30000.0            | -                 |
| DAC (High-Z)     | 18220.5              | 14962.5            | 15000.0            | 6550.4            |
| Inter-instrument | 36440.0              | 29925.0            | 30000.0            | 6550.4            |

## Using Digital I/O

On Moku:Delta and Moku:Go, a slot Input, Output or both may be routed to the Digital I/O block. In this case, the `Input` or `Output` signal should be interpreted as a 16-bit `std_logic_vector` or equivalent. Each bit of this vector corresponds to a digital I/O pin in the obvious way, i.e. `InputX(0)` contains the current logical value of DIO Pin 1, driving DIO Pin 16 is done by assigning a value to `OutputX(15)` (where `Input/OutputX` are the slot signals you've routed to the DIO block in the Multi-instrument Mode builder).

The Digital I/O block does not have automatic detection of driving sources, the I/O direction for each pin must be manually configured. On the MiM Configuration screen, click the Digital I/O block and set each pin's desired driving direction.

:::tip Driving an Input
If you attempt to drive a value to a pin configured as an input, that action is silently ignored. If you read in from a pin that's configured as an output, the operation succeeds and simply gives you the current logical value of the pin.
:::

There is an [example available](https://github.com/liquidinstruments/moku-examples/tree/main/mcc/Basic/DIO) for more information. Digital I/O is also used to drive the sync signals in the [VGA Example](https://github.com/liquidinstruments/moku-examples/tree/main/mcc/Advanced/VGA_Display)
