# Simple Boxcar Averager

## Overview

The boxcar averager instrument takes two inputs: a signal input and a gate (trigger) input. It starts integrating the input signal for a certain number of samples after receiving a trigger, then sends it to a secondary integrator to average a certain number of triggers events before sending it to the output.

This simple design is written directly in VHDL with hard-coded parameters. For a more featureful version that requires HDL Coder, see [HDL Coder Boxcar](https://github.com/liquidinstruments/moku-examples/tree/main/mcc/HDLCoder/hdlcoder_boxcar).

## Getting Started

### Signals and Settings

| Port | Use |
| -- | -- |
| Input A | Signal |
| Input B | Trigger |
| Output A | Average Out |
| Output B | Not Used |
