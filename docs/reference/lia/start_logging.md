---
additional_doc: The LIA contains an embedded Data Logger that can be started with this function. Returns name of the log file. By default all the log files are saved to persist directory.
description: "Start the data logging session to file. "
method: post
name: start_logging
parameters:
- default: 60
  description: Duration to log for
  name: duration
  param_range: null
  type: integer
  unit: Seconds
- default: ''
  description: Optional file name prefix
  name: file_name_prefix
  param_range: null
  type: string
  unit: null
- default: ''
  description: Optional comments to be included
  name: comments
  param_range: null
  type: string
  unit: null
- default: false
  description: Pass as true to stop any existing session and begin a new one
  name: stop_existing
  param_range: null
  type: boolean
  unit: null
  warning: Passing true will kill any existing data logging session with out any warning. Use with caution.
summary: start_logging
group: Embedded Data Logger
---

<headers/>

::: warning Caution
To ensure a complete data logging session, it is recommended to track the progress using [logging_progress](logging_progress.md).
:::

<parameters/>

Log files can be downloaded to local machine using [download_files](../static/download.md)

