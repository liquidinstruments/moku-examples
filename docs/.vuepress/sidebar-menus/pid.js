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
        ["/reference/pid/set_hysteresis", "set_hysteresis"],
        ["/reference/pid/set_input_offset", "set_input_offset"],
        ["/reference/pid/set_output_offset", "set_output_offset"],
        ["/reference/pid/summary", "summary"],
        {
            title: 'Logger',
            collapsable: true,
            children: [
                ["/reference/embedded/edl/start_logging", "start_logging"],
                ["/reference/embedded/edl/stop_logging", "stop_logging"],
                ["/reference/embedded/edl/logging_progress", "logging_progress"]
            ]
        },
        {
            title: 'Monitors',
            collapsable: true,
            children: [
                ["/reference/embedded/eos/get_data", "get_data"],
                ["/reference/embedded/eos/set_monitor", "set_monitor"],
                ["/reference/embedded/eos/set_timebase", "set_timebase"],
                ["/reference/embedded/eos/set_trigger", "set_trigger"],
            ]
        },
    ]
};