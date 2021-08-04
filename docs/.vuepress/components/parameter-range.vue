<script>
export default {
  functional: true,
  props: ["range"],
  render(h, { props }) {
    if (typeof props.range === "object") {
      var children = [];
      children.push(title(h, ""));
      children.push(h("table", { class: "parameter-range-table" }, [
        type_and_range(h, "Moku:Go", props.range.mokugo, "tr-go"),
        type_and_range(h, "Moku:Pro", props.range.mokupro, "tr-pro")
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
  text = "allowed values";
  return h("span", { class: "parameter-range-title" }, text);
}

function type_heading(h, text) {
  return h("span", { class: "parameter-range-header" }, text);
}

function type_and_range(h, title, text, className) {
  return h("tr",{class: className}, [
    h("td", title),
    h("td", text)
  ]);
}

</script>

<style>

.param-range-container{
display:grid;
}

.parameter-range-title {
  padding-right: 0.5em;
  font-weight:600;
  text-decoration: underline;
}

.parameter-range-header{
  font-weight:600;
  padding-right:0.25em;
  content:":"
}
.parameter-range-header:after,
.parameter-range-title:after
{
  content:":"
}

.parameter-range-table{
  border:none;
}

.tr-go td{
    background-color:#e7c000;
    color:#fff;
    font-weight:450;
}

</style>
