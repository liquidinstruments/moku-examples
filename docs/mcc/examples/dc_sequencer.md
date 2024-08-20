# DC Sequencer

A customer required a system that output a series of DC values, stepping between them based on a trigger input. Some AWG units can segment their memory and transition between segments in this way, but it is a complicated process given each segment contains a single value.

Using MCC, a simple sequencer was written with custom schmitt trigger logic and a pre-defined list of DAC output values. The debounced trigger was output from the logic as well, to be used downstream.

The full code is available on [Gitlab](https://gitlab.com/liquidinstruments/cloud-compile/examples/-/tree/main/dc_sequencer). The schmitt trigger levels are configurable through a Control register, the DC value sequence is hard-coded.
