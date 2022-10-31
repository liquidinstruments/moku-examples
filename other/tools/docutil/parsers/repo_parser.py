import os

from yaml import safe_load


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


class RepoParser:

    @staticmethod
    def param_differences(local_methods, valid, spec_ops):
        differences = {}
        for m in valid:
            local_def = [p[1] for p in local_methods if p[0] == m][0]
            spec_def = \
                [p['parameters'] for p in spec_ops if p['name'] == m][
                    0]
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
                param_names = [n['name'] for n in params or []]
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
        return instr_methods
