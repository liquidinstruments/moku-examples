import json
import pathlib


class SpecParser:
    def __init__(self, spec_path):
        if spec_path and pathlib.Path(spec_path).exists():
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
                    v["name"] = k
                    parameters.append(v)
        return dict(parameters=parameters,
                    summary=path_spec[method].get("summary"),
                    method=method,
                    description=path_spec[method].get("description"),)

    def get_operations_for_group(self, group, supports_slot=False):
        group_idx = 3 if supports_slot else 2
        paths = self.spec.get('paths')
        operations = []
        for p in paths.keys():
            split_path = p.split('/')
            if split_path[group_idx] == group:
                method_definition = dict(name=split_path[-1])
                method_definition.update(self.parse(paths[p]))
                operations.append(method_definition)
        return operations

    def get_instruments(self):
        paths = self.spec.get('paths')
        return set([p.split('/')[2] for p in paths.keys()])
