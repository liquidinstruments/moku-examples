const mokuPropAdmin = require('./sidebar-menus/moku')
const powerSuppliesAdmin = require('./sidebar-menus/powersupplies')
const awgAdmin = require('./sidebar-menus/awg')
const fraAdmin = require('./sidebar-menus/fra')
const wgAdmin = require('./sidebar-menus/wavegen')
const specanAdmin = require('./sidebar-menus/specan')
const laAdmin = require('./sidebar-menus/logicanalyzer')
const dlAdmin = require('./sidebar-menus/datalogger')
const pidAdmin = require('./sidebar-menus/pid')
const oscAdmin = require('./sidebar-menus/oscilloscope')
const pmAdmin = require('./sidebar-menus/phasemeter')
const staticAdmin = require('./sidebar-menus/static')
const miscAdmin = require('./sidebar-menus/misc')

// const dirTree = require('directory-tree');
// const path = require('path');

// const projets = dirTree(path.join(__dirname, '../projets'), {extensions:/\.md/});



module.exports = {
    title: 'Moku API',
    description: 'A demo documentation using VuePress',
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
            { text: 'API Reference', link: '/reference/' },
            {
                text: 'Examples',
                ariaLabel: 'Examples Menu',
                items: [
                    { text: 'Python', link: '/examples/python/' },
                    { text: 'Matlab', link: '/examples/matlab/' }
                ]
            },
            { text: 'Forum', link: 'https://forum.liquidinstruments.com/' },
            { text: 'Support', link: 'https://www.liquidinstruments.com/support/contact/' },
        ],
        sidebar: {
            '/reference': [
                mokuPropAdmin,
                powerSuppliesAdmin,
                awgAdmin,
                dlAdmin,
                fraAdmin,
                laAdmin,
                oscAdmin,
                pidAdmin,
                specanAdmin,
                wgAdmin,
                pmAdmin,
                staticAdmin,
                miscAdmin,
            ],
            '/examples': "auto",
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