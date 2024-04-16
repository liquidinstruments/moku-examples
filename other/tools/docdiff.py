import glob
import json
import os
import sys
from pprint import pprint

from yaml import safe_load

# Define map between spec path and the instrument directory name,
# this essentially compares the methods containing path with
# methods defined in that particular directory
INSTRUMENTS_MAP = dict(phasemeter="phasemeter", fra="fra",
                       logicanalyzer="logicanalyzer", lockinamp="lia",
                       waveformgenerator="waveformgenerator",
                       spectrumanalyzer="specan",
                       oscilloscope="oscilloscope",
                       datalogger="datalogger", firfilter="fir",
                       digitalfilterbox="dfb",
                       pidcontroller="pid", awg="awg")

IGNORE_METHODS = ["set_register", "get_register", "get_stream_header"]

# Vuepress docs are organized in a well defined directory structure,
# each instrument may have,
#   instrument/
#       *.md #All of core instrument methods
#       eos/ # Embedded oscilloscope methods
#       edl/ # Embedded Datalogger methods
#       Getters.md # All the getters are defined in a single file

# Now, when comparing the spec with actual codebase, diff tool should
# be looking at all the above mentioned directories along with
# parameter definitions in each file


class SpecParser:
    def __init__(self, spec_path):
        if spec_path and os.path.exists(spec_path):
            with open(spec_path, 'r') as _reader:
                data = _reader.read()
                self.spec = json.loads(data)
                components = self.spec.get('components')
                if components:
                    self.schemas = components.get('schemas')
                    print('Spec validated successfully...')
                else:
                    raise Exception("No components in spec")
        else:
            raise Exception("Empty or invalid spec path.")

    def _get_request_ref(self, desc):
        for k, v in desc.items():
            if isinstance(v, dict):
                return self._get_request_ref(v)
            if k == '$ref':
                schema_path = v.split('/')
                reference = schema_path[len(schema_path) - 1]
                if reference in self.schemas.keys():
                    return self.schemas[reference]

    def parse(self, path_spec):
        method = list(path_spec.keys())[0]
        parameters = []
        if 'requestBody' in path_spec[method].keys():
            req_ref = self._get_request_ref(
                path_spec[method].get('requestBody'))
            if req_ref:
                for k, v in req_ref.get('properties', {}).items():
                    parameters.append(k)
        return dict(parameters=parameters)

    def get_operations_for_group(self, group):
        paths = self.spec.get('paths')
        operations = []
        for p in paths.keys():
            if len(p.split('/')) == 5 and p.split('/')[3] == group:
                name = p.split('/')[4]
                if name not in IGNORE_METHODS:
                    method_definition = dict(name=name)
                    method_definition.update(self.parse(paths[p]))
                    operations.append(method_definition)
        return operations

    def get_instruments(self):
        paths = self.spec.get('paths')
        return set([p.split('/')[2] for p in paths.keys()])

    @staticmethod
    def param_differences(local_methods, valid, spec_ops):
        differences = {}
        for m in valid:
            local_def = [p[1] for p in local_methods if p[0] == m][0]
            spec_def = [p['parameters'] for p in spec_ops if p['name'] == m][0]
            param_diff = list(set(spec_def).difference(local_def))
            if param_diff:
                differences[m] = param_diff
        return differences

    @staticmethod
    def get_methods_from_getters(target):
        _getter_f = os.path.join(target, 'getters.md')
        result = []
        if os.path.exists(_getter_f):
            with open(os.path.join(_getter_f), 'r') as _r:
                data = _r.read().split('---')[1]
                getters = safe_load(data).get('getters', [])
                for g in getters:
                    g_params = g.get('parameters', [])
                    params = [n['name'] for n in g_params]
                    result.append((g['summary'], params))
                return result

    def _read_method_and_parameters(self, target):
        result = []
        _fs = [x for x in os.listdir(target) if
               os.path.isfile(os.path.join(target, x)) and
               not x.startswith('README')]
        for _f in _fs:
            _m_name = _f.split('.md')[0]
            with open(os.path.join(target, _f), 'r') as _r:
                data = _r.read().split('---')
                params = safe_load(data[1]).get('parameters', [])
                param_names = [n['name'] for n in params]
                result.append((_m_name, param_names))
        getters = self.get_methods_from_getters(target)
        if getters:
            result.extend(getters)
        return result

    def get_methods_from_local_repo(self, instr_dir):
        instr_methods = []
        if os.path.exists(instr_dir):
            # read core files...
            instr_methods.extend(
                self._read_method_and_parameters(instr_dir))
            # read edl files...
            edl_dir = os.path.join(instr_dir, 'edl')
            if os.path.exists(edl_dir):
                instr_methods.extend(
                    self._read_method_and_parameters(edl_dir))
            # read eos files...
            eos_dir = os.path.join(instr_dir, 'eos')
            if os.path.exists(eos_dir):
                instr_methods.extend(
                    self._read_method_and_parameters(eos_dir))
            # read streaming files...
            es_dir = os.path.join(instr_dir, 'streaming')
            if os.path.exists(es_dir):
                instr_methods.extend(
                    self._read_method_and_parameters(es_dir))
            return instr_methods
        raise Exception("Cannot find instrument directory")

    def compare(self):
        result = {}
        for k, v in INSTRUMENTS_MAP.items():
            result[k] = {}
            required = self.get_operations_for_group(k)
            required_methods = [x['name'] for x in required]
            docs_path = os.path.join('../../docs/reference', v)
            local_methods = self.get_methods_from_local_repo(
                docs_path)
            doc_methods = [x[0] for x in local_methods]
            missing_methods = list(
                set(required_methods).difference(doc_methods))
            missing_params = self.param_differences(
                local_methods,
                set(required_methods).intersection(doc_methods),
                required)
            if missing_methods:
                result[k]["missing_methods"] = missing_methods

            if missing_params:
                result[k]["missing_methods"] = missing_params

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

    s = SpecParser(spec)
    pprint(s.compare())
