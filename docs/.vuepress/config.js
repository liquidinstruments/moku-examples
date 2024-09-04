const mimAdmin = require('./sidebar-menus/mim')
const mokuPropAdmin = require('./sidebar-menus/moku')
const getChildren = require('./getChildren').getChildren
const externalClockAdmin = require('./sidebar-menus/ext_clk')
const powerSuppliesAdmin = require('./sidebar-menus/powersupplies')
const staticAdmin = require('./sidebar-menus/static')

module.exports = {
    title: 'Moku API',
    dest: 'public/' + (process.env.BUILD_DIR || ''),
    base: process.env.BASE_PATH || '/',
    description:
        'Documentation for the Moku Scripting API for Python and MATLAB',
    head: [
        [
            'link',
            {
                rel: 'preconnect',
                href: 'https://rsms.me/',
            },
        ],
        [
            'link',
            {
                rel: 'stylesheet',
                href: 'https://rsms.me/inter/inter.css',
            },
        ],
    ],
    themeConfig: {
        logo: '/assets/img/logo.svg',
        smoothScroll: false,
        nav: [
            // {
            //     text: 'Test dropdown',
            //     ariaLabel: 'Dropdown Menu',
            //     items: [
            //         { text: 'One', link: '/one/' },
            //         { text: 'Two', link: '/two/' },
            //     ],
            // },
            { text: 'Rest API', link: '/api/' },
            { text: 'Moku Cloud Compile', link: '/mcc/' },
            { text: 'Moku CLI', link: '/cli/' },
            { text: 'Forum', link: 'https://forum.liquidinstruments.com/' },
            {
                text: 'Support',
                link: 'https://www.liquidinstruments.com/support/contact/',
            },
        ],
        sidebar: {
            '/api/': [
                {
                    title: 'Moku Scripting API',
                    collapsable: false,
                    initialOpenGroupIndex: -1,
                    sidebarDepth: 1,
                    children: [['/api/', 'Overview']],
                },
                {
                    title: 'Getting Started',
                    collapsable: false,
                    initialOpenGroupIndex: -1,
                    sidebarDepth: 1,
                    children: [
                        ['/api/getting-started/', 'Overview'],
                        ['/api/getting-started/starting-python', 'Python'],
                        ['/api/getting-started/starting-matlab', 'MATLAB'],
                        ['/api/getting-started/starting-labview', 'LabVIEW'],
                        ['/api/getting-started/starting-curl', 'cURL'],
                        ['/api/getting-started/starting-mim', 'MiM'],
                        ['/api/getting-started/starting-other', 'Other'],
                        ['/api/getting-started/ip-address', 'IP Address'],
                        [
                            '/api/getting-started/download-bitstreams',
                            'Download Bitstreams',
                        ],
                    ],
                },
                {
                    title: 'API Reference',
                    collapsable: false,
                    initialOpenGroupIndex: -1,
                    children: [['/api/reference/', 'Overview']],
                },
                {
                    title: 'Core Functions',
                    collapsable: false,
                    initialOpenGroupIndex: -1,
                    children: [
                        mokuPropAdmin,
                        externalClockAdmin,
                        staticAdmin,
                        powerSuppliesAdmin,
                        mimAdmin,
                    ],
                },
                {
                    title: 'Instruments',
                    collapsable: false,
                    initialOpenGroupIndex: -1,
                    children: [
                        getChildren(
                            'Arbitrary Waveform Generator',
                            'reference/awg',
                        ),
                        getChildren('Cloud Compile', 'reference/cloudcompile'),
                        getChildren('Datalogger', 'reference/datalogger'),
                        getChildren('Digital Filter Box', 'reference/dfb'),
                        getChildren('FIR FIlter', 'reference/fir'),
                        getChildren(
                            'Frequency Response Analyzer',
                            'reference/fra',
                        ),
                        getChildren('Laser Lock Box', 'reference/llb'),
                        getChildren('LockIn Amplifier', 'reference/lia'),
                        getChildren(
                            'Logic Analyzer',
                            'reference/logicanalyzer',
                        ),
                        getChildren('Oscilloscope', 'reference/oscilloscope'),
                        getChildren('Phasemeter', 'reference/phasemeter'),
                        getChildren('PID Controller', 'reference/pid'),
                        getChildren('Spectrum Analyzer', 'reference/specan'),
                        getChildren(
                            'Time & Frequency Analyzer',
                            'reference/tfa',
                        ),
                        getChildren(
                            'Waveform Generator',
                            'reference/waveformgenerator',
                        ),
                    ],
                },
                {
                    title: 'Examples',
                    collapsable: false,
                    initialOpenGroupIndex: -1,
                    children: [
                        getChildren('Python', 'examples/python'),
                        getChildren('MatLab', 'examples/matlab'),
                        getChildren(
                            'Other Languages',
                            'examples/other-languages',
                        ),
                    ],
                },
            ],
            '/mcc/': [
                {
                    title: 'Moku Cloud Compile',
                    collapsable: false,
                    initialOpenGroupIndex: -1,
                    children: [['/mcc/', 'Overview']],
                },
                {
                    title: 'Features',
                    collapsable: false,
                    initialOpenGroupIndex: -1,
                    children: [
                        ['/mcc/slots', 'Multi-Instrument and Slots'],
                        ['/mcc/wrapper', 'Custom Wrapper'],
                        ['/mcc/support', 'Moku Library'],
                        ['/mcc/controls', 'Control Registers'],
                        ['/mcc/deploying', 'Deploying Your Design'],
                    ],
                },
                {
                    title: 'Examples',
                    collapsable: false,
                    initialOpenGroupIndex: -1,
                    children: [
                        ['/mcc/examples/', 'Basic Examples'],
                        ['/mcc/examples/servo', 'Servo Driver'],
                        ['/mcc/examples/dio', 'Digital I/O'],
                        ['/mcc/examples/dc_sequencer', 'DC Sequencer'],
                        ['/mcc/examples/event_counter', 'Event Counter'],
                        [
                            '/mcc/examples/hdlcoder_sqrt',
                            'Simulink™ sqrt of signals',
                        ],
                    ],
                },
            ],
            '/cli/': [
                {
                    title: 'Moku CLI',
                    collapsable: false,
                    initialOpenGroupIndex: 0,
                    children: [
                        ['/cli/', 'Overview'],
                        ['/cli/moku-cli', 'Usage'],
                    ],
                },
            ],
        },
    },
    markdown: {
        lineNumbers: true,
    },
    plugins: [
        '@vuepress/back-to-top',
        'versioning',
        [
            'vuepress-plugin-code-copy',
            {
                align: 'top',
                color: '#48b8e7',
            },
        ],
    ],
    configureWebpack: {
        resolve: {
            alias: {
                '@mcc': '../../../linked-mcc-examples',
            },
        },
    },
}
