import json
import os
import yaml

INSTRUMENTS_MAP = dict(phasemeter="phasemeter", fra="fra",
                       logicanalyzer="logicanalyzer", lockinamp="lia",
                       waveformgenerator="waveformgenerator",
                       spectrumanalyzer="specan", oscilloscope="oscilloscope",
                       datalogger="datalogger", firfilter="firfilter",
                       pidcontroller="pid", awg="awg")


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
        spec = path_spec[method]
        parameters = []
        if 'requestBody' in spec.keys():
            req_ref = self._get_request_ref(spec.get('requestBody'))
            if req_ref:
                for k, v in req_ref.get('properties', {}).items():
                    parameters.append(k)
        return dict(parameters=parameters)

    def get_operations_for_group(self, group):
        paths = self.spec.get('paths')
        operations = []
        for p in paths.keys():
            if p.split('/')[2] == group:
                method_definition = dict(name=p.split('/')[3])
                method_definition.update(self.parse(paths[p]))
                operations.append(method_definition)
        return operations

    def get_instruments(self):
        paths = self.spec.get('paths')
        return set([p.split('/')[2] for p in paths.keys()])

    @staticmethod
    def param_differences(docs_path, methods, definitions):
        differences = {}
        for m in methods:
            spec_definition = [x for x in definitions if x['name'] == m][0]
            with open(os.path.join(docs_path, m + '.md'), 'r') as _r:
                data = _r.read().split('---')
                params = [n['name'] for n in yaml.safe_load(data[1]).get('parameters')]
                param_diff = list(set(spec_definition['parameters']).difference(params))
                if param_diff:
                    differences[m] = param_diff
        return differences

    def compare(self):
        result = {}
        for k, v in INSTRUMENTS_MAP.items():
            result[k] = {}
            required = self.get_operations_for_group(k)
            docs_path = os.path.join('../../docs/reference', v)
            if os.path.exists(docs_path):
                doc_methods = [x.split('.')[0] for x in os.listdir(docs_path)]
            else:
                doc_methods = []
            required_methods = [x['name'] for x in required]
            result[k]["missing_methods"] = list(set(required_methods).difference(doc_methods))
            result[k]["missing_params"] = self.param_differences(docs_path,
                                                                 set(required_methods).intersection(doc_methods),
                                                                 required)

        return result


s = SpecParser('/Users/Sashi/workspace/rest-documentation/other/specs/spec_557.json')
print(json.dumps(s.compare()))
