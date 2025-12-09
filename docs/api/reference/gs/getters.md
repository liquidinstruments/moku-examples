---
additional_doc: Along with regular configuration methods, Gigabit Streamer supports following getter functions. Utilize the summary function to readback all settings.
name: getters
description: Gigabit Streamer - getter functions
getters:
- summary: get_send_status
  description: Get current sending status
- summary: get_receive_status
  description: Get current receiving status, including the number of channels, overflow count, packet count, sampling rate and underflow count
- summary: get_frontend
  description: Get the input impedance, coupling, and range for given input channel
  parameters:
  - default: null
    description: Target channel
    name: channel
    param_range:
     mokudelta: 1, 2, 3, 4
    type: integer
    unit: null
---

<headers/>
<getters/>
