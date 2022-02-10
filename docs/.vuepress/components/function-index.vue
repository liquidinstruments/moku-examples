<template>
    <div>
        <h2>Functions</h2>
        <!-- <section v-for="(pages, group) in functionGroups" :key="group"> -->
        <section v-for="group in orderedGroupNames" :key="group">
            <h3>{{ group }}</h3>
            <ul>
                <li v-for="page in functionGroups[group]" :key="page.path">
                    <RouterLink :to="link=page.path">{{page.frontmatter.name}}</RouterLink>: {{ page.frontmatter.description }}</li>
            </ul>
        </section>
    </div>
</template>

<style lang="stylus" scoped>

</style>

<script>
export default {
    computed: {
        functionGroups () {
            // Currently only works if the embedding page URL ends in '/'
            const myPath = this.$page.path
            const children = this.$site.pages.filter((p) => (p.path.startsWith(myPath) && p.path.length > myPath.length));

            let funcs = {}

            for (let i = 0; i < children.length; i++) {
                const child = children[i];
                const group = child.frontmatter.group || 'Core'
                if (!funcs[group]) {
                    funcs[group] = []
                }

                funcs[group].push(child)
            }

            for (const group in funcs) {
                funcs[group].sort((a, b) => (a.frontmatter.name > b.frontmatter.name))
            }

            return funcs
        },

        orderedGroupNames() {
            const fg = this.functionGroups
            return Object.keys(fg).sort()
        }
    }
}
</script>
