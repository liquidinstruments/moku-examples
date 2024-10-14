# Servo Driver

This module takes an analog input (perhaps from a PID Controller in an adjacent slot) and converts it to a 50Hz pulse train, suitable for position control of common hobby servo motors.

The top-level module creates two counter blocks, one that governs the rate of the pulses and the other that ticks at the time resolution of the pulse width. The counters divide the input clock by `2^(EXPONENT) / Increment`, i.e. `2^24 / 3` giving roughly 50Hz from the Pro's 312.5MHz clock for the overall pulse timer, and `2^15 / 107` giving roughly 2048 steps within a 2ms pulse.

The output is nominally a digital signal, with the bit values corresponding to `high` and `low` voltages defined as constants, `0x000` and `0x7FF` corresponding to zero and maximum DAC output range respectively.

This example is [available on Gitlab](https://gitlab.com/liquidinstruments/cloud-compile/examples/-/tree/main/servo).

:::tip Clock Rate
The example is configured for Moku:Pro's 312.5MHz clock. The divider ratios should be changed before you attempt to run this on another platform.
:::

## Top-Level Module

<<< @/docs/api/moku-examples/mcc/servo/Top.vhd

## Counter/Timer Module

<<< @/docs/api/moku-examples/mcc/servo/Counter.vhd
