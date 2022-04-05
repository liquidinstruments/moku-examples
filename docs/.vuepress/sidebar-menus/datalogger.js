module.exports = {
    title: 'Datalogger',
    collapsable: true,
    children: [
        ["/reference/datalogger/", "Overview"],
        ["/reference/datalogger/disable_channel", "disable_channel"],
        ["/reference/datalogger/generate_waveform", "generate_waveform"],
        ["/reference/datalogger/logging_progress", "logging_progress"],
        ["/reference/datalogger/set_acquisition_mode", "set_acquisition_mode"],
        ["/reference/datalogger/set_frontend", "set_frontend"],
        ["/reference/datalogger/set_samplerate", "set_samplerate"],
        ["/reference/datalogger/start_logging", "start_logging"],
        ["/reference/datalogger/stop_logging", "stop_logging"],
        ["/reference/datalogger/summary", "summary"],
        {
            title: 'Getters',
            collapsable: true,
            children: [
                ["/reference/datalogger/get_acquisition_mode", "get_acquisition_mode"],
                ["/reference/datalogger/get_frontend", "get_frontend"],
                ["/reference/datalogger/get_samplerate", "get_samplerate"],
            ]
        }
    ]
};