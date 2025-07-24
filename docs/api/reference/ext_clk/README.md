---
prev: false
---

# External Reference Clock

These functions provide access to the 10MHz external reference clock available on the Moku.

These functions work without an instrument deployed and do not disrupt the operation of any instrument that is running.

## Blended Clock

The Moku:Delta blended clock offers an advanced solution to precise timing.

Users can choose between a 10MHz and 100MHz external clock, changes to the external reference clock will take effect when your Moku:Delta is restarted. These options are connected via the same BNC port on back panel.

Moku:Delta can receive precision timing from the atomic clocks utilized in the various Global Navigation Satellite Systems (GNSSs). The 1 PPS signal utilizes an external GPS-disciplined oscillator, such as, another Moku:Delta.

In the absence of any external input, Moku:Delta takes advantage of a bespoke internal phase reference that is internally generated.

<function-index/>
