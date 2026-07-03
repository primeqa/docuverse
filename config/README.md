# DocUVerse configuration layout

Config files are looked up by a shared resolver
(`docuverse/utils/config_resolver.py`). Engines request paths relative to a
config base dir — e.g. `servers/milvus_servers.json` — and the resolver
searches these tiers in order, first match wins:

1. `$DOCUVERSE_HOME/<rel_path>` — operator-level override.
2. `~/.docuverse/<rel_path>` — per-user override.
3. `./config/<rel_path>` — this directory, categorized layout (preferred).
4. `./config/<basename>` — legacy flat layout. Works, but emits a one-shot
   `DeprecationWarning` asking you to move the file.
5. Packaged defaults shipped inside the wheel (`docuverse/config_defaults/`),
   so `pip install docuverse` works without a checkout.
6. `FileNotFoundError` listing every path tried.

## Categorized layout

```
config/
  servers/         # connection registries: which Milvus/Elastic servers exist
    milvus_servers.json
    elastic_servers.json
  engines/         # per-backend defaults: index/search parameter presets
    milvus_default_config.yaml
    elastic_config.json
  <name>_data_format.yml   # data/query format templates (referenced directly
                           # by experiment YAMLs, so they stay at the top level)
```

Recipe presets (the `docuverse presets` / `SearchEngine.from_preset` names)
live in the package itself under `docuverse/presets/recipes/` — dump one with
`docuverse presets dump milvus-dense > my-recipe.yaml` and edit from there.
Experiment configs live under `experiments/`.

## Server registries

`servers/*.json` maps a short name to a connection spec:

```json
{
  "localhost": {"host": "localhost", "port": 19530},
  "prod":      {"host": "milvus.example.com", "port": 19530, "secure": true}
}
```

A config's `server:` field then takes one of three forms:

- a registry name: `server: localhost`
- an embedded (Milvus-Lite) file: `server: "file:./.docuverse/milvus.db"`
- an inline dict: `server: {host: milvus.example.com, port: 19530}`

## Non-interactive runs

Library prompts ("cache file exists, read?", "recreate index?") read stdin.
Set `DOCUVERSE_NONINTERACTIVE=1` in CI, notebooks-via-nbconvert, or batch jobs
to auto-answer every prompt with its default.
