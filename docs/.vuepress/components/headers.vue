<template>
  <div>
    <h1>{{ $frontmatter.summary }} 
      <warn-go v-if="$frontmatter.available_on === 'mokugo'"/>
      <warn-pro v-if="$frontmatter.available_on === 'mokupro'"/>
    </h1>
    <h2>{{ $frontmatter.description }}</h2>
    <span class="additional-doc">{{ $frontmatter.additional_doc }}</span>
  </div>
</template>

<style scoped>
.additional-doc {
  margin-bottom: 1em;
}
</style>
