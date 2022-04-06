module.exports = {
    title: 'FIR Filter Builder',
    collapsable: true,
    children: [
        ["/reference/fir/", "Overview"],
        ["/reference/fir/enable_input", "enable_input"],
        ["/reference/fir/enable_output", "enable_output"],
        ["/reference/fir/set_by_frequency", "set_by_frequency"],
        ["/reference/fir/set_by_gain_and_section", "set_by_gain_and_section"],
        ["/reference/fir/set_by_gain", "set_by_gain"],
        ["/reference/fir/set_control_matrix", "set_control_matrix"],
        ["/reference/fir/set_defaults", "set_defaults"],
        ["/reference/fir/set_frontend", "set_frontend"],
        ["/reference/fir/set_hysteresis", "set_hysteresis"],
        ["/reference/fir/set_input_offset", "set_input_offset"],
        ["/reference/fir/set_output_offset", "set_output_offset"],
        ["/reference/fir/summary", "summary"],
        {
            title: 'Logger',
            collapsable: true,
            children: [
                ["/reference/fir/edl/start_logging", "start_logging"],
                ["/reference/fir/edl/stop_logging", "stop_logging"],
                ["/reference/fir/edl/logging_progress", "logging_progress"]
            ]
        },
        {
            title: 'Monitors',
            collapsable: true,
            children: [
                ["/reference/fir/eos/enable_rollmode", "enable_rollmode"],
                ["/reference/fir/eos/get_data", "get_data"],
                ["/reference/fir/eos/save_high_res_buffer", "save_high_res_buffer"],
                ["/reference/fir/eos/set_acquisition_mode", "set_acquisition_mode"],
                ["/reference/fir/eos/set_hysteresis", "set_hysteresis"],
                ["/reference/fir/eos/set_timebase", "set_timebase"],
                ["/reference/fir/eos/set_trigger", "set_trigger"],
            ]
        },
    ]
};