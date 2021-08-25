---
additional_doc: null
description: Sets trigger source and parameters.
method: post
name: set_trigger
available_on: "mokugo"
parameters:

- default: null
  description: Trigger Source
  name: source
  param_range: Pin1, Pin2, Pin3, Pin4, Pin5, Pin6, Pin7, Pin8, Pin9, Pin10, Pin11,
    Pin12, Pin13, Pin14, Pin15, Pin16
  type: string
  unit: null
- default: Edge
  description: Trigger type
  name: type
  param_range: Edge, Pulse
  type: string
  unit: null
- default: Auto
  description: Trigger mode
  name: mode
  param_range: Auto, Normal, Single
  type: string
  unit: null
- default: Rising
  description: Which edge to trigger on (edge mode only)
  name: edge
  param_range: Rising, Falling, Both
  type: string
  unit: null
- default: Positive
  description: Trigger pulse polarity
  name: polarity
  param_range: Positive, Negative
  type: string
  unit: null
- default: GreaterThan
  description: Trigger pulse width condition (pulse mode only)
  name: width_condition
  param_range: GreaterThan, LessThan
  type: string
  unit: null
- default: 0.0001
  description: Trigger width
  name: width
  param_range: 26e-3 to 10
  type: number
  unit: Seconds
- default: 1
  description: The number of trigger events to wait for before triggering
  name: nth_event
  param_range: 0 to 65535
  type: integer
  unit: null
- default: 0
  description: The duration to hold off Oscilloscope trigger post trigger event.
  name: holdoff
  param_range: 1e-9 to 10
  type: number
  unit: Seconds
- default: true
  description: Disable all implicit conversions and coercions.
  name: strict
  param_range: null
  type: boolean
  unit: null
summary: set_trigger
---





<headers/>
<parameters/>