---
next: servo
---

# Getting Started

These code snippets are fully self-contained examples, each of which can be built in Moku Cloud Compile and achieves some basic task.

## More Examples

Many more examples are available on [Gitlab](https://gitlab.com/liquidinstruments/cloud-compile/examples/).

## On This Page

[[toc]]

## The Basic Component Structure

All Moku Cloud Compile top-level blocks are architectures of [CustomWrapper](../wrapper.md).

<<< @/docs/api/moku-examples/mcc/basic/Top.vhd

## Control Registers

The Custom Wrapper has Inputs, Outputs and [Control Registers](../controls.md). The input and output routing is determined in the Multi-instrument Mode (MiM) configuration screen. They can be connected to ADCs and DACs respectively, but can also be attached to other slots in order to pre- or post-process instrument data.

The control registers have 32-bit values that can be changed through the MCC instrument screen on the Moku: application. Here we interpret 16 bits of each of the first two control registers as a (signed) DC voltage to output from the MCC instrument. For example, if the MiM configuration routes the MCC slot outputs to the DACs, this is a simple programmable DC supply.

<<< @/docs/api/moku-examples/mcc/regs_basic/Top.vhd

Control register bits can also be used to enable and disable features.

<<< @/docs/api/moku-examples/mcc/reg_gate/Gate.vhd

## Some arithmetic

Basic arithmetic is available in the VHDL language, but note that this is purely combinatorial so can run in to timing errors.

<<< @/docs/api/moku-examples/mcc/adder/Adder.vhd

## Instantiate a DSP

For more complex arithmetic, it's common to explicitly instantiate a DSP block in the FPGA. The [`Moku.Support.ScaleOffset`](../support.html#scaleoffset) entity conveniently packages a DSP block with all the settings configured to compute the common `Z = X * Scale + Offset` operation, with the output properly clipped to prevent under/overflow.

<<< @/docs/api/moku-examples/mcc/dsp/DSP.vhd

## Moku Library

The [Moku Library](../support.md) contains many useful helper functions and components, like the `ScaleOffset` block used above. For example, the `clip` function clips a signal to a defined bit range, gracefully handling saturation.

<<< @/docs/api/moku-examples/mcc/clip/Clip.vhd
