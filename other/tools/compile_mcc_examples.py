"""
This script iterates through mcc_examples_in and 
compiles a set of markdown files to use in documentation

Author: Sashi Yalamarthi
Date: 16/10/2024
"""

from pathlib import Path
from pprint import pprint

mcc_examples_in = Path(
    "/home/sashi/PycharmProjects/rest-documentation/docs/api/moku-examples/mcc"
)
mcc_examples_out = Path("/home/sashi/PycharmProjects/rest-documentation/docs/mcc/examples")

root = [d for d in list(mcc_examples_in.glob("*")) if d.is_dir()]

for d in root:
    name = mcc_examples_out.joinpath(d.name.lower())
    examples = [d for d in list(d.glob("*")) if d.is_dir()]
    with open(f"{name}.md", "w+") as _md:
        for ex in examples:
            md_def = [x for x in list(ex.glob("*.md"))]
            if md_def:
                _md.write(md_def[0].read_text())
                _md.write("\n\n")
