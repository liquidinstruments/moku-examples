const mokuPropAdmin = require('./sidebar-menus/moku')
const getChildren = require('./getChildren').getChildren
const powerSuppliesAdmin = require('./sidebar-menus/powersupplies')
const staticAdmin = require('./sidebar-menus/static')
const { children } = require('./sidebar-menus/moku')

// const dirTree = require('directory-tree');
// const path = require('path');

// const projets = dirTree(path.join(__dirname, '../projets'), {extensions:/\.md/});



module.exports = {
    title: 'Moku API',
    dest: "public/" + (process.env.BUILD_DIR || ""),
    base: process.env.BASE_PATH || "/",
    description: 'Documentation for the Moku Scripting API for Python and MATLAB',
    head: [
        [
            "link",
            {
                rel: "stylesheet",
                href: "https://use.typekit.net/dnm2hhw.css"
            }
        ]
    ],
    themeConfig: {
        logo: '/assets/img/logo.svg',
        smoothScroll: true,
        sidebarDepth: 3,
        nav: [
            { text: 'API Home', link: '/' },
            { text: 'API Reference', link: '/reference/' },
            {
                text: 'Examples',
                ariaLabel: 'Examples Menu',
                items: [
                    { text: 'Python', link: '/examples/python/' },
                    { text: 'Matlab', link: '/examples/matlab/' },
                    { text: 'Other Languages', link: '/examples/other-languages/' }
                ]
            },
            { text: 'Forum', link: 'https://forum.liquidinstruments.com/' },
            { text: 'Support', link: 'https://www.liquidinstruments.com/support/contact/' },
        ],
        sidebar: {
            '/reference': [
                ['/', "Documentation Home"],
                ['/reference/', "API Reference Home"],
                {
                    title: "Core Functions",
                    collapsable: false,
                    initialOpenGroupIndex: -1,
                    children: [
                        mokuPropAdmin,
                        staticAdmin,
                        powerSuppliesAdmin,
                    ]
                },

                {
                    title: "Instruments",
                    collapsable: false,
                    initialOpenGroupIndex: -1,
                    children: [
                        getChildren("Arbitrary Waveform Generator", "awg"),
                        getChildren("Datalogger", "datalogger"),
                        getChildren("Digital Filter Box", "dfb"),
                        getChildren("FIR FIlter", "fir"),
                        getChildren("Frequency Response Analyzer", "fra"),
                        getChildren("LockIn Amplifier", "lia"),
                        getChildren("Logic Analyzer", "logicanalyzer"),
                        getChildren("Oscilloscope", "oscilloscope"),
                        getChildren("Phasemeter", "phasemeter"),
                        getChildren("PID Controller", "pid"),
                        getChildren("Spectrum Analyzer", "specan"),
                        getChildren("Waveform Generator", "waveformgenerator"),
                    ]
                }
            ],
            '/examples': [
                '/',
                '/examples/',
                '/examples/python/',
                '/examples/matlab/',
                '/examples/other-languages/'
            ],
            '/': [{
                    title: 'Getting Started',
                    collapsable: false,
                    children: [
                        'starting-python',
                        'starting-matlab',
                        'starting-labview',
                        'starting-curl',
                        'starting-other',
                        'ip-address'
                    ]
                },
                ['/reference/', "API Reference"],
                ['/examples/', "Examples"]
            ]
        }
    },
    markdown: {
        lineNumbers: true
    },
    plugins: ['@vuepress/back-to-top', 'versioning', ['vuepress-plugin-code-copy', {
        align: "top",
        color: "#48b8e7"
    }]]
};