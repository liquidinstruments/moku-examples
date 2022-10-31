import glob
import os
import sys
from pprint import pprint

from docutil import INSTRUMENTS_MAP, NON_INSTRUMENTS_MAP
from docutil.parsers import SpecParser, RepoParser


def param_differences(ops_in_spec, ops_in_repo, ops_to_check):
    differences = {}
    for op in ops_to_check:
        local_params = [r[1] for r in ops_in_repo if r[0] == op][0]
        spec_params = [p['name'] for p in
                       [s['parameters'] for s in ops_in_spec
                        if s['name'] == op][0]]
        missing = list(set(spec_params).difference(local_params))
        unexpected = list(set(local_params).difference(spec_params))
        if missing or unexpected: differences[op] = {}
        if missing: differences[op]["missing"] = missing
        if unexpected: differences[op]["unexpected"] = unexpected
    return differences


def compare_group(spec_name, local_name, has_slot):
    result = {}
    operations = spec_parser.get_operations_for_group(
        spec_name, has_slot)
    op_names = [o['name'] for o in operations]
    docs_path = os.path.join('../../docs/reference', local_name)
    local_operations = repo_parser.get_methods_from_local_repo(
        docs_path)
    local_op_names = [lo[0] for lo in local_operations]
    result["missing_methods"] = list(
        set(op_names).difference(local_op_names))
    result["missing_params"] = param_differences(
        operations,
        local_operations,
        set(op_names).intersection(local_op_names))
    return result


def compare():
    result = {}
    for spec_name, local_name in INSTRUMENTS_MAP.items():
        result[spec_name] = compare_group(spec_name, local_name, 1)

    for spec_name, local_name in NON_INSTRUMENTS_MAP.items():
        result[spec_name] = compare_group(spec_name, local_name, 0)

    return result


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
    repo_parser = RepoParser()
    pprint(compare())
