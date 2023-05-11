<template>
  <div class="bitstream-container">
    <input
      class="bitstream-textbox"
      type="text"
      v-model="firmwareVer"
      placeholder="Enter firmware version"
    />
    <button class="bitstream-button" @click.prevent="constructBitstreamsURL">
      Bitstreams
    </button>
    <button class="bitstream-button" @click="constructChecksumURL">
      Checksum
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
    formattedUrl() {
      if (this.firmwareVer === "") {
        alert("Enter a valid firmware version number");
        return;
      }
      return (
        "https://updates.liquidinstruments.com/static/mokudata-" +
        this.firmwareVer
      );
    },

    constructBitstreamsURL() {
      const baseURL = this.formattedUrl();
      if (baseURL) window.open(baseURL + ".tar.gz");
    },

    constructChecksumURL() {
      const baseURL = this.formattedUrl() + ".md5";
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
