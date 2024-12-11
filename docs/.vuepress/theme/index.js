/**
 * @param {import('@vuepress/types').Context['pages']} pages
 * @param {string[]} pathArray
 */
function getPageTitle(pages, pathArray) {
    const path = '/' + pathArray.join('/') + '/'
    return pages.find((page) => page.regularPath === path)?.title
}

/**
 * @type {import('@vuepress/types').Theme}
 */
module.exports = {
    extend: '@vuepress/theme-default',

    /**
     * Generate extra metadata to be attached to each page. Currently used for
     * adding extra search data
     */
    extendPageData($page) {
        if (!$page.frontmatter) return

        const path = $page.regularPath.split('/')
        path.shift() // Remove leading slash

        const hasPageTitle = !!$page.title
        const isApiPage = path[0] === 'api'

        // Generate a page title if one was not provided. This is only used by
        // search and the html <title> tag
        // Would default to the h1 title, if one could be determined
        if (!hasPageTitle) {
            $page.title =
                $page.frontmatter.summary || $page.frontmatter.name || undefined

            // For api pages, append the instrument to them
            if ($page.title && isApiPage) {
                const instrumentPage = path.slice(0, 3)
                $page.title += ` | ${getPageTitle($page._context.pages, instrumentPage) || path[2]}`
            }
        }
    },
}

