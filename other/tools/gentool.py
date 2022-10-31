import glob
import sys
from pprint import pprint
import pathlib
import jinja2

from docutil.parsers import SpecParser

doc_template = jinja2.Template("""
---
additional_doc: null
description: {{description}}
method: {{method}}
name: {{name}}
parameters:
{% for p in parameters %}
- default: {{p.default}}
  description: {{p.description}}
  name: {{p.name}}
  param_range: {{p.range}}
  type: {{p.type}}
  unit: {{p.unit}}
{% endfor %}
summary: {{summary}}
---


<headers/>

<parameters/>

### Examples

<code-group>
<code-block title="Python">
```python

```
</code-block>

<code-block title="MATLAB">
```matlab

```
</code-block>

<code-block title="cURL">
```bash
# You should create a JSON file with the data content rather than passing
# arguments on the CLI as the lookup data is necessarily very large
$: cat request.json
{
 
}
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data @request.json\
        
```
</code-block>

</code-group>

### Sample response

""", trim_blocks=True, lstrip_blocks=True)


def generate(group, has_slot, write_to):
    write_to = pathlib.Path(write_to).joinpath(group)
    write_to.mkdir(exist_ok=True)
    operations = spec_parser.get_operations_for_group(group, has_slot)
    for o in operations:
        with open(write_to.joinpath(o["name"] + ".md"), 'w+') as _w:
            _w.write(doc_template.render(o))


if __name__ == '__main__':
    if len(sys.argv) == 1:
        specs = glob.glob('../specs/spec*.json')
        specs.sort()
        spec = specs[-1]
        print(f"No spec given, choosing {spec}")
    elif len(sys.argv) == 2:
        spec = sys.argv[1]
    else:
        raise Exception("Either give me a spec or leave me be")

    spec_parser = SpecParser(spec)
    pprint(generate("mim", False, "/home/sashi/PycharmProjects/rest-documentation/docs/reference"))
