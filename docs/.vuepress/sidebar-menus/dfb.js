module.exports = {
    title: 'Digital filter box',
    collapsable: true,
    children: [
        ["/reference/dfb/", "Overview"],
        ["/reference/dfb/enable_output", "enable_output"],
        ["/reference/dfb/set_control_matrix", "set_control_matrix"],
        ["/reference/dfb/set_custom_filter", "set_custom_filter"],
        ["/reference/dfb/set_defaults", "set_defaults"],
        ["/reference/dfb/set_filter", "set_filter"],
        ["/reference/dfb/set_frontend", "set_frontend"],
        ["/reference/dfb/set_input_gain", "set_input_gain"],
        ["/reference/dfb/set_input_offset", "set_input_offset"],
        ["/reference/dfb/set_monitor", "set_monitor"],
        ["/reference/dfb/set_output_gain", "set_output_gain"],
        ["/reference/dfb/set_output_offset", "set_output_offset"],
        ["/reference/dfb/summary", "summary"],
        ["/reference/dfb/getters", "Getters"],
        {
            title: 'Logger',
            collapsable: true,
            children: [
                ["/reference/dfb/edl/start_logging", "start_logging"],
                ["/reference/dfb/edl/stop_logging", "stop_logging"],
                ["/reference/dfb/edl/logging_progress", "logging_progress"]
            ]
        },
        {
            title: 'Monitors',
            collapsable: true,
            children: [
                ["/reference/dfb/eos/enable_rollmode", "enable_rollmode"],
                ["/reference/dfb/eos/get_data", "get_data"],
                ["/reference/dfb/eos/save_high_res_buffer", "save_high_res_buffer"],
                ["/reference/dfb/eos/set_acquisition_mode", "set_acquisition_mode"],
                ["/reference/dfb/eos/set_hysteresis", "set_hysteresis"],
                ["/reference/dfb/eos/set_timebase", "set_timebase"],
                ["/reference/dfb/eos/set_trigger", "set_trigger"],
                ["/reference/dfb/eos/getters", "Getters"],
            ]
        },
    ]
};