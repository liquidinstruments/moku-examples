const eosDefinitions = require('./eos')
const edlDefinitions = require('./edl')

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
        edlDefinitions,
        eosDefinitions,
    ]
};
