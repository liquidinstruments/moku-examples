<template>
  <div v-if="$frontmatter.getters.length">
    <div v-for="g in $frontmatter.getters">
      <h2>
        {{ g.summary }}
      </h2>
      <span class="description">{{ g.description }}</span>
      <p class="additional-doc">{{ g.additional_doc }}</p>
      <div v-if="g.parameters && g.parameters.length">
        <h3>Parameters</h3>
        <section class="parameters-section">
          <div v-for="p in g.parameters">
            <div class="param-container" :id="'param-' + p.name">
              <div class="Parameter-module--padded--1K5Kq">
                <div class="Title-module--title--16rrq">
                  <label
                    ><a :href="'#param-' + p.name" data-display-anchor="true">{{
                      p.name
                    }}</a>
                  </label>
                  <span class="parameter-types"> {{ p.type }}</span>
                  <span class="parameter-obligation" v-if="p.default == null">
                    required</span
                  >
                </div>
                <div class="parameter-description">
                  <div>
                    {{ p.description }}
                  </div>
                </div>
                <div
                  class="Detail-module--details--3v0Zg"
                  v-if="p.default !== null"
                >
                  <code v-if="p.default === ''">default: ""</code>
                  <code v-else>default: {{ p.default }}</code>
                </div>
                <div
                  class="Detail-module--details--3v0Zg"
                  v-if="p.param_range !== null"
                >
                  <parameter-range :range="p.param_range"></parameter-range>
                </div>
                <div
                  class="Detail-module--details--3v0Zg"
                  v-if="p.unit !== null"
                >
                  <code>units: {{ p.unit }}</code>
                </div>

                <div v-if="typeof p.warning !== 'undefined'">
                  <div class="warning custom-block">
                    <p class="custom-block-title">Caution</p>
                    <p>
                      {{ p.warning }}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>
      </div>
      <hr />
    </div>
  </div>
</template>

<style scoped>
.description {
  font-size: 1rem;
}

.additional-doc {
  margin-bottom: 2em;
}

h2 {
  border-bottom: none;
}

h3 {
  margin-bottom: 0;
}

.parameter-description {
  padding-top: 0.5em;
  padding-bottom: 0.5em;
}

.parameter-warning {
  color: #6b5900;
  background-color: #fff7d0;
  padding: 0.05em 1em;
  font-weight: 450;
}
.Detail-module--details--3v0Zg {
  font-size: 15px;
  margin-top: 0.5rem;
  display: flex;
}

.Title-module--title--16rrq label {
  font-weight: 900;
  vertical-align: baseline;
  margin-right: 0.5rem;
}

.Endpoint-module--reference--3acDy a[data-display-anchor="true"]:hover::before {
  content: "#";
  font-family: lato, sans-serif;
  position: absolute;
  margin-left: -0.7em;
  color: #c3d1d9;
}

.parameter-types {
  display: inline-block;
  color: #002947;
  font-size: 13px;
  background-color: #fbfbfc;
  padding: 0.2em 0.5em;
  border-radius: 2px;
  border: 1px solid #e9eef3;
}

.parameters-section {
  display: block;
  width: 100%;
  min-width: 348px;
  margin-bottom: 2rem !important;
  position: relative;
}

.param-container {
  position: relative;
  background-color: #fff;
  padding: 1em;
}

.Title-module--optional--eykfy {
  position: absolute;
  top: 0;
  right: 0;
  font-size: 11.2px;
  border-radius: 0 0 0 4px;
  color: #607079;
  padding: 0.25rem 0.5rem;
  font-weight: 400;
  border-color: #dde6ed;
  border-style: solid;
  border-width: 0 0 1px 1px;
}

.parameter-obligation {
  position: absolute;
  font-size: 11.2px;
  color: red;
  padding: 0.25rem 0.5rem;
  font-weight: 400;
}
</style>
