<script>
export default {
  functional: true,
  props: ["range"],
  render(h, { props }) {
    if (typeof props.range === "object") {
      var children = [];
      children.push(title(h, ""));
      children.push(h("table", { class: "parameter-range-table" }, [
        type_and_range(h, "Moku:Lab", props.range.mokulab, "tr-lab"),
        type_and_range(h, "Moku:Go", props.range.mokugo, "tr-go"),
        type_and_range(h, "Moku:Pro", props.range.mokupro, "tr-pro"),
      ]));
      return h("div", { class: "param-range-container"  }, [h("code", children)]);
    } 
    else {
      return h("div", { class: "param-range-container" }, 
              [h("code", [title(h, ""), h("span", props.range)])]);
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
