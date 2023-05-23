<template>
  <div>
    <h1>
      {{ $frontmatter.summary }} <beta-flag v-if="$frontmatter.mark_as_beta === true"/>
      <Badge v-if="$frontmatter.available_on" :text="$frontmatter.available_on" type="warn" />
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
