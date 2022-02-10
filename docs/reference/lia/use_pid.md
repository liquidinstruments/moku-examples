---
additional_doc: There is a single PID block internally that may be connected to either channel. The PID parameters can be configured with `set_by_frequency`.
description: Enables and disables a PID controller on either the Main or Aux outputs
method: post
name: use_pid
parameters:
- default: null
  description: Which channel, if any, to PID control
  name: channel
  param_range: Main, Aux, Off
  type: string
  unit: null
- default: true
  description: Disable all implicit conversions and coercions.
  name: strict
  param_range: null
  type: boolean
  unit: null
summary: use_pid
group: PID Controller
---

<headers/>
<parameters/>
