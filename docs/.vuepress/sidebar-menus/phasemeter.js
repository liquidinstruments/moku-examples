module.exports = {
    title: 'Phasemeter',
    collapsable: true,
    children: [
        ["/reference/phasemeter/", "Overview"],
        ["/reference/phasemeter/disable_freewheeling", "disable_freewheeling"],
        ["/reference/phasemeter/disable_output", "disable_output"],
        ["/reference/phasemeter/generate_output", "generate_output"],
        ["/reference/phasemeter/get_acquisition_speed", "get_acquisition_speed"],
        ["/reference/phasemeter/get_auto_acquired_frequency", "get_auto_acquired_frequency"],
        // ["/reference/phasemeter/get_data", "get_data"],
        ["/reference/phasemeter/set_acquisition_speed", "set_acquisition_speed"],
        ["/reference/phasemeter/set_auto_reset", "set_auto_reset"],
        ["/reference/phasemeter/set_defaults", "set_defaults"],
        ["/reference/phasemeter/set_frontend", "set_frontend"],
        ["/reference/phasemeter/set_phase_wrap", "set_phase_wrap"],
        ["/reference/phasemeter/set_pm_loop", "set_pm_loop"],
        ["/reference/phasemeter/summary", "summary"],
        {
            title: 'Logger',
            collapsable: true,
            children: [
                ["/reference/phasemeter/edl/start_logging", "start_logging"],
                ["/reference/phasemeter/edl/stop_logging", "stop_logging"],
                ["/reference/phasemeter/edl/logging_progress", "logging_progress"]
            ]
        },
    ]
};