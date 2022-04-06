module.exports = {
    title: 'Digital filter box',
    collapsable: true,
    children: [{
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