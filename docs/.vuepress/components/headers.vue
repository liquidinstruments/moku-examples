<template>
  <div>
    <h1>
      {{ $frontmatter.summary }} <beta-flag v-if="$frontmatter.mark_as_beta === true"/>
      <Badge v-if="$frontmatter.deprecated" text="deprecated" type="error" />
      <Badge v-if="$frontmatter.available_on" :text="$frontmatter.available_on" type="warn" />
    </h1>
    <span class="description">{{ $frontmatter.description }}</span>
    <p class="additional-doc">{{ $frontmatter.additional_doc }}</p>
    <div v-if="$frontmatter.deprecated_msg" class="custom-block danger">
      <p class="custom-block-title">Deprecated</p> 
      <p v-html="formattedMarkdownContent"></p>
    </div>
  </div>
</template>
<script>
import markdownIt from 'markdown-it';
export default {
  computed: {
    formattedMarkdownContent() {
      const md = new markdownIt();
      return md.render(this.$frontmatter.deprecated_msg);
    },
  },
}
</script>

<style scoped>
.description {
  font-size: 1.4rem;
}

.additional-doc {
  margin-bottom: 2em;
}
</style>
