<template>
  <div class="bitstream-container">
    <input
      class="bitstream-textbox"
      type="text"
      v-model="firmwareVer"
      placeholder="Enter firmware version"
    />
    <button class="bitstream-button" @click.prevent="constructBitstreamsURL">
      Download Bitstreams
    </button>
  </div>
</template>

<script>
export default {
  data() {
    return {
      firmwareVer: "",
    };
  },
  methods: {
    constructBitstreamsURL() {
      if (this.firmwareVer === "") {
        alert("Enter a valid firmware version number");
        return;
      }
      const baseURL =
        "https://updates.liquidinstruments.com/static/mokudata-" +
        this.firmwareVer +
        ".tar.gz";
      window.open(baseURL);
    },
  },
};
</script>

<style>
.bitstream-container {
  margin: 1em;
}
.bitstream-textbox {
  padding: 8px 12px;
  font-size: 14px;
  border: 1px solid #ced4da;
  border-radius: 4px;
  outline: none;
}

.bitstream-button {
  padding: 8px 16px;
  font-size: 14px;
  background-color: #48b8e7;
  color: #fff;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}

.bitstream-button:hover {
  background-color: #3a9ac6;
}
</style>
