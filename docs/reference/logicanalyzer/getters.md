---
additional_doc: Along with regular configuration methods, Logic Analyzer supports following getter functions.
name: getters
description: Logic Analyzer -  getter functions
getters: 
- summary: get_decoder
  description: Gets the decoder configuration for the given ID.
- summary: get_pattern_generator
  description: Gets the configuration for a given pattern generator ID.
- summary: get_trigger
  description: Gets the current trigger configuration.
  additional_doc: Response includes edge, holdoff, mode, nth_event, polarity, source, type, width, width_condition
- summary: get_timebase
  description: Gets the span and offset for the configured timebase
---
<headers/>
<getters/>
