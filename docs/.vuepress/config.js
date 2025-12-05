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
            { text: 'REST API', link: '/api/' },
            { text: 'Moku Compile', link: '/mc/' },
            { text: 'Moku CLI', link: '/cli/' },
            { text: 'Moku Neural Network', link: '/mnn/' },
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
                        getChildren('Moku Compile', 'reference/custominstrument'),
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
                        getChildren('Neural Network', 'reference/nn'),
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
                        ['examples/', 'Overview'],
                        getChildren('Python', 'moku-examples/python-api'),
                        getChildren('MatLab', 'moku-examples/matlab-api'),
                        getChildren(
                            'Other Languages',
                            'moku-examples/other-language-api',
                        ),
                    ],
                },
            ],
            '/mc/': [
                {
                    title: 'Moku Compile',
                    collapsable: false,
                    initialOpenGroupIndex: -1,
                    children: [['/mc/', 'Overview']]
                },
                {               
                    title: 'Getting Started',
                    collapsable: false,
                    initialOpenGroupIndex: -1,
                    children: [['/mc/getting-started/cloud', 'Moku Compile'],
                               ['/mc/getting-started/deploying', 'Deploying Your Design']
                            ],
                },
                {
                    title: 'Features',
                    collapsable: false,
                    initialOpenGroupIndex: -1,
                    children: [
                        ['/mc/slots', 'Slot Resources and Rates'],
                        ['/mc/wrapper', 'Wrapper for Moku Compile'],
                        ['/mc/support', 'Moku Library'],
                        ['/mc/ipcore', 'IP Core Support'],
                        ['/mc/io', 'Input and Output'],
                        ['/mc/controls', 'Control Registers'],
                        ['/mc/statusregs', 'Status Registers'],
                        ['/mc/browser', 'Bitstream Browser']
                    ],
                },
                {
                    title: 'Examples',
                    collapsable: false,
                    initialOpenGroupIndex: -1,
                    children: [
                        ['/mc/examples/template', 'Template'],
                        ['/mc/examples/basic', 'Basic'],
                        ['/mc/examples/moderate', 'Moderate'],
                        ['/mc/examples/advanced', 'Advanced'],
                        ['/mc/examples/hdlcoder', 'HDL Coder'],
                    ],
                },
            ],
            '/cli/': [
                {
                    title: 'Moku CLI',
                    collapsable: false,
                    initialOpenGroupIndex: -1,
                    children: [
                        ['/cli/', 'Overview'],
                    ],
                },
                {
                    title: 'Usage',
                    collapsable: false,
                    initialOpenGroupIndex: -1,
                    children: [
                        ['/cli/instrument', 'instrument'],
                        ['/cli/firmware', 'firmware'],
                        ['/cli/feature', 'feature'],
                        ['/cli/convert', 'convert'],
                        ['/cli/download', 'download'],
                        ['/cli/list', 'list'],
                        ['/cli/files', 'files'],
                        ['/cli/license', 'license'],
                        ['/cli/proxy', 'proxy'],
                        ['/cli/stream', 'stream'],
                        ['/cli/advanced', 'Advanced Usage'],
                    ],
                },
            ],
            '/mnn/': [
                {
                    title: 'Moku Neural Nework',
                    collapsable: false,
                    initialOpenGroupIndex: 0,
                    children: [['/mnn/', 'Overview']],
                },
                {
                    title: 'The LinnModel class',
                    collapsable: false,
                    initialOpenGroupIndex: -1,
                    children: [
                        ['/mnn/linnmodel-class/linnmodel', 'LinnModel class'],
                    ],
                },
                {
                    title: 'Examples',
                    collapsable: false,
                    initialOpenGroupIndex: -1,
                    children: [
                        ['/mnn/examples/', 'Overview'],
                        ['/mnn/examples/Anomaly_detection', 'Anomaly Detection'],
                        ['/mnn/examples/Autoencoder', 'Autoencoder'],
                        ['/mnn/examples/Classification', 'Classification'],
                        ['/mnn/examples/Emitter_control', 'Emitter Control'],
                        ['/mnn/examples/Identity_network', 'Identity Network'],
                        ['/mnn/examples/Signal_ID', 'Signal Identifier'],
                        ['/mnn/examples/Simple_sine', 'Simple Sine wave'],
                        ['/mnn/examples/Sum', 'Weighted Sum'],
                    ],
                },
            ],
        },
    },
    markdown: {
        lineNumbers: true,
        extendMarkdown: (md) => {

            const fence = md.renderer.rules.fence
            md.renderer.rules.fence = (...args) => {
                const [tokens, idx] = args
                const token = tokens[idx]

                // Map .vhd to vhdl
                if (token.info === 'vhd') {
                    token.info = 'vhdl'
                }

                if (token.info === 'v') {
                    token.info = 'verilog'
                }

                if (token.info === 'm') {
                    token.info = 'matlab'
                }

                return fence(...args)
            }
        },
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
                '@mc': '../api/moku-examples/mc',
            },
        },
    },
}
