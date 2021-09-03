<template>
    <p
        class="action"
    >
        <NavLink
          class="action-button"
          :item="actionLink"
        />
    </p>
</template>

<script>
import NavLink from '@theme/components/NavLink.vue'

export default {
    name: "ActionButton",

    components: { NavLink },
    
    props: {
        text: {
            type: String,
            default: 'text'
        },
        link: {
            type: String,
            default: ''
        }
    },

    computed: {

        actionLink () {
            return {
                link: this.link,
                text: this.text
            }
        }
    }
}
</script>

<style lang="stylus">
.action
    margin 1.8rem auto
.action-button
    text-align center
    display inline-block
    font-size 1.2rem
    color #fff
    background-color $accentColor
    padding 0.8rem 1.6rem
    border-radius 4px
    transition background-color .1s ease
    box-sizing border-box
    border-bottom 1px solid darken($accentColor, 10%)
    &:hover
        background-color lighten($accentColor, 10%)
</style>
