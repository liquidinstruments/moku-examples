<script>
export default {
  functional: true,
  props: ["range"],
  render(h, { props }) {
    if (typeof props.range === "object") {
      const rows = []

      if (props.range.mokulab != null) {
        rows.push(type_and_range(h, "Moku:Lab", props.range.mokulab, "tr-lab"));
      }

      if (props.range.mokugo != null) {
        rows.push(type_and_range(h, "Moku:Go", props.range.mokugo, "tr-go"));
      }

      if (props.range.mokupro != null) {
        rows.push(type_and_range(h, "Moku:Pro", props.range.mokupro, "tr-pro"));
      }

      if (props.range.mokudelta != null) {
        rows.push(type_and_range(h, "Moku:Delta", props.range.mokudelta, "tr-delta"));
      }

      if (props.range.gigabitstreamer != null) {
        rows.push(type_and_range(h, "Gigabit Streamer", props.range.gigabitstreamer, "tr-gs"));
      }

      if (props.range.gigabitstreamerplus != null) {
        rows.push(type_and_range(h, "Gigabit Streamer+", props.range.gigabitstreamerplus, "tr-gsp"));
      }

      const children = [];
      children.push(title(h, ""));
      children.push(
        h("table", { class: "parameter-range-table" }, rows)
      );

      return h("div", { class: "param-range-container"  }, [
        h("code", children),
      ]);
    } else {
      return h("div", { class: "param-range-container" }, [
        h("code", [title(h, ""), h("span", props.range)]),
      ]);
    }
  },
};

function title(h, text) {
  text = "allowed values:";
  return h("span", { class: "parameter-range-title" }, text);
}

function type_and_range(h, title, text, className) {
  return h("tr",{class: className}, [
    h("td", title),
    h("td", text)
  ]);
}

</script>

<style lang="stylus">
.param-range-container
  display: grid

.parameter-range-title
  padding-right: 0.5em

.parameter-range-table
  border:none
  margin: 1em


</style>
