module.exports = {
    title: 'Lock-In Amplifier',
    collapsable: true,
    children: [
        ["/reference/lia/", "Overview"],
        ["/reference/lia/enable_rollmode", "enable_rollmode"],
        ["/reference/lia/pll_reacquire", "pll_reacquire"],
        ["/reference/lia/set_aux_output", "set_aux_output"],
        ["/reference/lia/set_by_frequency", "set_by_frequency"],
        ["/reference/lia/set_defaults", "set_defaults"],
        ["/reference/lia/set_demodulation", "set_demodulation"],
        ["/reference/lia/set_filter", "set_filter"],
        ["/reference/lia/set_frontend", "set_frontend"],
        ["/reference/lia/set_gain", "set_gain"],
        ["/reference/lia/set_hysteresis", "set_hysteresis"],
        ["/reference/lia/set_outputs", "set_outputs"],
        ["/reference/lia/set_pll_bandwidth", "set_pll_bandwidth"],
        ["/reference/lia/set_polar_mode", "set_polar_mode"],
        ["/reference/lia/summary", "summary"],
        ["/reference/lia/use_pid", "use_pid"],
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