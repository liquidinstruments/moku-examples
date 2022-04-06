module.exports = {
    title: 'PID Controller',
    collapsable: true,
    children: [
        ["/reference/pid/", "Overview"],
        ["/reference/pid/enable_input", "enable_input"],
        ["/reference/pid/enable_output", "enable_output"],
        ["/reference/pid/set_by_frequency", "set_by_frequency"],
        ["/reference/pid/set_by_gain_and_section", "set_by_gain_and_section"],
        ["/reference/pid/set_by_gain", "set_by_gain"],
        ["/reference/pid/set_control_matrix", "set_control_matrix"],
        ["/reference/pid/set_defaults", "set_defaults"],
        ["/reference/pid/set_frontend", "set_frontend"],
        ["/reference/pid/set_input_offset", "set_input_offset"],
        ["/reference/pid/set_monitor", "set_monitor"],
        ["/reference/pid/set_output_offset", "set_output_offset"],
        ["/reference/pid/summary", "summary"],
        {
            title: 'Logger',
            collapsable: true,
            children: [
                ["/reference/pid/edl/start_logging", "start_logging"],
                ["/reference/pid/edl/stop_logging", "stop_logging"],
                ["/reference/pid/edl/logging_progress", "logging_progress"]
            ]
        },
        {
            title: 'Monitors',
            collapsable: true,
            children: [
                ["/reference/pid/eos/enable_rollmode", "enable_rollmode"],
                ["/reference/pid/eos/get_data", "get_data"],
                ["/reference/pid/eos/save_high_res_buffer", "save_high_res_buffer"],
                ["/reference/pid/eos/set_acquisition_mode", "set_acquisition_mode"],
                ["/reference/pid/eos/set_hysteresis", "set_hysteresis"],
                ["/reference/pid/eos/set_timebase", "set_timebase"],
                ["/reference/pid/eos/set_trigger", "set_trigger"],
            ]
        },
    ]
};