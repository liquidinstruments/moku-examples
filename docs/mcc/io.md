# Input and Output

The MCC design, as defined in the [Entity wrapper](./wrapper.md), has four 16-bit inputs and four 16-bit outputs (`InputA-D` and `OutputA-D` respectively). These signals can be routed to come to/from physical ADC/DAC channels on the hardware, Digital I/O where supported, or from other instruments in the Multi-instrument configuration.

## Clk and Reset

|       | Moku:Go     | Moku:Pro    |
| ----- | ----------- | ----------- |
| Clk   | 31.25MHz    | 312.5MHz    |
| Reset | Active-High | Active-High |

## Inputs and Outputs

Input and Output ports on the wrapper can either be:

-   Connected to other ports through the Moku Application's block diagram
-   Permanently connected to particular ADCs or DACs, or
-   Not connected

|         | Moku:Go       | Moku:Pro      |
| ------- | ------------- | ------------- |
| InputA  | Block Diagram | Block Diagram |
| InputB  | Block Diagram | Block Diagram |
| InputC  | ADC In 1      | Not Connected |
| InputD  | ADC In 2      | Not Connected |
| OutputA | Block Diagram | Block Diagram |
| OutputB | Block Diagram | Block Diagram |
| OutputC | Block Diagram | Not Connected |
| OutputD | Not Connected | Not Connected |

All Inputs and Outputs are 16-bit signed values. When the port is externally connected to Digital I/O, the signed 16-bit values should be interpreted simply as a 16-bit standard logic vector.

## Analog I/O Scaling

The bits-to-volts scaling of ADCs, DACs and between instruments is as follows. This table assumes no ADC attenuation and no DAC gain is configured.

| Source/Sink      | Moku:Go bits/volt | Moku:Pro bits/volt |
| ---------------- | ----------------- | ------------------ |
| ADC              | 409.4             | 2270.02            |
| DAC (50R)        | -                 | 29925.0            |
| DAC (High-Z)     | 6550.4            | 14962.5            |
| Inter-instrument | 13100.8           | 29925.0            |

:::warning Pass-through
Note that scaling is different depending on source. This can lead to a number of unexpected effects, for example the trivial "passthrough" instrument `OutputA <= InputA` actually scales the signal down by 16x if passed from ADC to DAC (High-Z) on Moku:Go (`6550.4 / 409.4`). This also means that the apparent signal amplitude changes depending whether the MCC design uses an ADC or another instrument as its source.
:::

## Using Digital I/O

On Moku:Go, a slot Input, Output or both may be routed to the Digital I/O block. In this case, the `Input` or `Output` signal should be interpretted as a 16-bit `std_logic_vector` or equivalent. Each bit of this vector corresponds to a digital I/O pin in the obvious way, i.e. `InputX(0)` contains the current logical value of DIO Pin 1, driving DIO Pin 16 is done by assigning a value to `OutputX(15)` (where `Input/OutputX` are the slot signals you've routed to the DIO block in the Multi-instrument Mode builder).

The Digital I/O block does not have automatic detection of driving sources, the I/O direction for each pin must be manually configured. On the MiM Configuration screen, click the Digital I/O block and set each pin's desired driving direction.

:::tip Driving an Input
If you attempt to drive a value to a pin configured as an input, that action is silently ignored. If you read in from a pin that's configured as an output, the operation succeeds and simply gives you the current logical value of the pin.
:::

There is an [example available](examples/dio.md) for more information. Digital I/O is also used to drive the sync signals in the [VGA Example](examples/vga_display.md)
