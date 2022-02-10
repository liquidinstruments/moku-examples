<template>
  <div>
    <h1>{{ $frontmatter.summary }} 
      <warn-go v-if="$frontmatter.available_on === 'mokugo'"/>
      <warn-pro v-if="$frontmatter.available_on === 'mokupro'"/>
    </h1>
    <span class="description">{{ $frontmatter.description }}</span>
    <p class="additional-doc">{{ $frontmatter.additional_doc }}</p>
  </div>
</template>

<style scoped>

.description {
  font-size: 1.4rem;
}

.additional-doc {
  margin-bottom: 2em;
}
</style>
