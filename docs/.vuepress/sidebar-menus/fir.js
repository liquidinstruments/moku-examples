const eosDefinitions = require('./eos')
const edlDefinitions = require('./edl')

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
        edlDefinitions,
        eosDefinitions
    ]
};