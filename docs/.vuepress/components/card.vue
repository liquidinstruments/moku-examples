<template>
    <component :is="wrapperTag" :href="link" v-bind="wrapperAttrs" class="card">
        <div class="card-header">
            <component
                :is="currentIcon"
                :size="24"
                color="#48b8e7"
                v-if="currentIcon"
            />
            <h3 v-if="title">{{ title }}</h3>
        </div>
        <slot> </slot>
        <span> Read the docs </span>
    </component>
</template>

<script>
import markdownIt from 'markdown-it'
import {
    PhArrowsLeftRight,
    PhTerminalWindow,
    PhCloudArrowUp,
} from 'phosphor-vue'

export default {
    name: 'card',
    methods: {
        formatMarkdown(data) {
            const md = new markdownIt()
            return md.render(data)
        },
    },
    components: {
        PhArrowsLeftRight,
        PhTerminalWindow,
        PhCloudArrowUp,
    },
    props: {
        link: {
            type: String,
            default: '',
        },
        icon: {
            type: String, // Define the type of the prop
            required: false, // Make this prop required
        },
        title: {
            type: String,
            default: '',
        },
        description: {
            type: String,
            default: 'description',
        },
    },
    computed: {
        currentIcon() {
            // Map the string prop to the corresponding component
            const icons = {
                mcc: 'PhCloudArrowUp',
                cli: 'PhTerminalWindow',
                api: 'PhArrowsLeftRight',
            }
            // Return the component name or null if not found
            return this.$options.components[icons[this.icon]] || null
        },
        wrapperTag() {
            return this.link ? 'a' : 'div'
        },
        wrapperAttrs() {
            return this.link ? { href: this.link } : {}
        },
    },
}
</script>

<style scoped lang="stylus">
.card {
    border-radius: 1.25rem;
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 1rem;
    border: 1px solid $borderColor;
    padding: 1.25rem;
    position: relative;
    background-color: rgba(255,255,255,70%);
}

.card-header {
    display: flex;
    flex-direction: row;
    align-items: center;
    gap: 1rem;
  }


h3 {
    border: none;
    font-size: 1rem;
    margin: 0;
    padding: 0;
}

 a.card:hover {
    text-decoration: none;
}

p {
    margin-block-start: 0.25rem;
    font-size: 0.9rem;
}

p:first-of-type {
    margin-top: 0;
}

p:last-of-type {
    margin-bottom: 0;
}

span {
    margin-top: auto;
    align-self: flex-start;
    border-bottom: 1px solid $accentColor;
    font-size: 0.9rem;
    font-weight: 600;
}
</style>
