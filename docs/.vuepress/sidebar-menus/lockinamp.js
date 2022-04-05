const eosDefinitions = require('./eos')
const edlDefinitions = require('./edl')

module.exports = {
    title: 'Lock-In Amplifier',
    collapsable: true,
    children: [
        ["/reference/lia/", "Overview"],
        ["/reference/lia/pll_reacquire", "pll_reacquire"],
        ["/reference/lia/set_aux_output", "set_aux_output"],
        ["/reference/lia/set_by_frequency", "set_by_frequency"],
        ["/reference/lia/set_defaults", "set_defaults"],
        ["/reference/lia/set_demodulation", "set_demodulation"],
        ["/reference/lia/set_filter", "set_filter"],
        ["/reference/lia/set_frontend", "set_frontend"],
        ["/reference/lia/set_gain", "set_gain"],
        ["/reference/lia/set_monitor", "set_monitor"],
        ["/reference/lia/set_outputs", "set_outputs"],
        ["/reference/lia/set_pll_bandwidth", "set_pll_bandwidth"],
        ["/reference/lia/set_polar_mode", "set_polar_mode"],
        ["/reference/lia/summary", "summary"],
        ["/reference/lia/use_pid", "use_pid"],
        edlDefinitions,
        eosDefinitions
    ]
};