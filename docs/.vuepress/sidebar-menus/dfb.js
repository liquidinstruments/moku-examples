const eosDefinitions = require('./eos')
const edlDefinitions = require('./edl')

module.exports = {
    title: 'Digital filter box',
    collapsable: true,
    children: [
        edlDefinitions,
        eosDefinitions,
    ]
};
