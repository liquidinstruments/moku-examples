# Event Counter

## Register and Parameter Definition

![Timing Diagram](@/docs/api/moku-examples/mcc/event_counter/waveform.png)

| | **Bits** ||
| **Register** | `31-16` | `15-0` |
| ------------ | :---------------------: | :-------------------: |
| Control0 | `t1:` clock cycles ||
| Control1 | `tpmax` clock cycles | `tpmin` clock cycles |
| Control2 | `mincount` count | `vpmin` ADC Bits |
