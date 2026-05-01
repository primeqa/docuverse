# Shared YAML config loader.
#
# Source this file, then:
#   config_file=$(find_config_arg "$@")
#   if [[ -n "$config_file" ]]; then
#       eval "$(load_yaml_config "$config_file")" || exit 1
#   fi
#
# Top-level YAML keys become bash variables of the same name:
#   - bool   -> "true" / "false"  (caller decides how to translate to a flag)
#   - list   -> bash array        (e.g. models: [a, b]  ->  models=(a b))
#   - scalar -> shell-quoted string

# Echo the value following the first `--config` flag in $@, if any.
find_config_arg() {
    while [[ $# -gt 0 ]]; do
        if [[ "$1" == "--config" ]]; then
            echo "$2"
            return 0
        fi
        shift
    done
}

# Read $1 as YAML and print bash assignments to stdout for `eval`.
# On parse errors, prints `exit 1`-bearing snippets so the caller's eval aborts.
load_yaml_config() {
    local config="$1"
    if [[ ! -f "$config" ]]; then
        echo "echo 'Config file not found: $config' >&2; exit 1"
        return 1
    fi
    python3 - "$config" <<'PYEOF'
import sys, shlex
try:
    import yaml
except ImportError:
    print('echo "PyYAML is required to read --config files (pip install pyyaml)" >&2; exit 1')
    sys.exit(0)
try:
    with open(sys.argv[1]) as f:
        cfg = yaml.safe_load(f) or {}
except Exception as e:
    print(f'echo {shlex.quote(f"Failed to parse YAML {sys.argv[1]}: {e}")} >&2; exit 1')
    sys.exit(0)
if not isinstance(cfg, dict):
    print('echo "YAML root must be a mapping (key: value pairs)" >&2; exit 1')
    sys.exit(0)
for k, v in cfg.items():
    if not k.replace('_', '').isalnum():
        print(f'echo {shlex.quote(f"Invalid YAML key (must be [A-Za-z0-9_]+): {k}")} >&2; exit 1')
        continue
    if isinstance(v, bool):
        print(f'{k}={"true" if v else "false"}')
    elif isinstance(v, list):
        items = ' '.join(shlex.quote(str(x)) for x in v)
        print(f'{k}=({items})')
    elif v is None:
        print(f'{k}=')
    else:
        print(f'{k}={shlex.quote(str(v))}')
PYEOF
}
