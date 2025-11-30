# Slot Resources and sampling rates

The FPGA resources are divided between slots and the supporting logic in the
platform surrounding the slots.  The table below summarizes the resources available
to a custom design in each slot.

|                    | Moku:Delta (XCZU47DR) | Moku:Delta (XCZU47DR) | Moku:Pro (ZU9EG) | Moku:Lab (ZC7020) | Moku:Lab (ZC7020) | Moku:Go (ZC7020) | Moku:Go (ZC7020) |
| ------------------ | --------------------: | --------------------: | ---------------: | ----------------: | ----------------: | ---------------: | ---------------: |
|                    |               3 Slots |               8 Slots |          4 Slots |           2 Slots |           3 Slots |          2 Slots |          3 Slots |
| Core Clock         |              312.5MHz |              312.5MHz |         312.5MHz |            125MHz |            125MHz |         31.25MHz |         31.25MHz |
| LUT                |                 50000 |                 20000 |            48400 |             19600 |             12000 |            20000 |            12000 |
| FF                 |                100000 |                 40000 |            96800 |             39200 |             24000 |            40000 |            24000 |
| BRAM (36K)         |                   100 |                    50 |              154 |                60 |                40 |               50 |               40 |
| DSP                |                   500 |                   200 |              432 |               100 |                60 |              100 |               60 |