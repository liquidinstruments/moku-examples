<template>
    <section class="card-group">
        <slot></slot>
    </section>
</template>

<script>
import markdownIt from 'markdown-it'

export default {
    name: 'card-group',
    methods: {
        formatMarkdown(data) {
            const md = new markdownIt()
            return md.render(data)
        },
    },
}
</script>

<style scoped lang="stylus">

.card-group {
    display: flex;
    flex-direction: column;
    gap: 2rem;

    @media screen and (min-width: 768px) {
        flex-direction: row;
    }
}
</style>
