# Provenance

`sherlock-osint-remastered` is an **independent derivative work** of the
[Sherlock Project](https://github.com/sherlock-project/sherlock).

It is **not affiliated with, endorsed by, or supported by** the Sherlock
Project or its maintainers. Do not report issues with this repository to
upstream, and do not report issues with upstream here.

## Upstream

| | |
| - | - |
| Project | Sherlock Project |
| Source | <https://github.com/sherlock-project/sherlock> |
| License | MIT |
| Copyright | © 2019 Sherlock Project |

Upstream's MIT license is retained in [LICENSE](LICENSE) alongside the
copyright for changes made in this repository. The original copyright notice
is preserved as the license requires.

## Inherited from upstream

- The site manifest (`sherlock_project/resources/data.json`) and its schema.
- Site iteration and result modelling (`sites.py`, `result.py`).
- The scan loop's argument surface and output formats (CSV, XLSX, TXT).
- Issue templates, code of conduct, and the regression/exclusion workflows.

## Added in this repository

None of the following exists upstream:

- Stealth-browser fetching via Playwright (`playwright_engine.py`).
- SQLite result persistence with schema migration (`database.py`).
- Bounded profile-content preparation (`content_extraction.py`).
- Two-pass local-model analysis: per-site structured extraction plus
  cross-site profile synthesis (`ai_engine.py`, `profile_synthesis.py`,
  `synthesis_pipeline.py`, `investigation_context.py`).
- A local AI provider integration and its setup flow (`ai_provider.py`,
  `ai_config.py`, `ai_setup.py`).
- The `rich`-based terminal reporter (`notify.py`).

## Services still consumed from upstream

These are deliberate runtime dependencies on upstream-operated resources, not
oversights. They are disclosed here so that operators know what this tool
talks to:

| Resource | Used for | Defined in |
| - | - | - |
| `https://data.sherlockproject.xyz` | Live site manifest | `sherlock_project/sites.py` |
| `.../sherlock/refs/heads/exclusions/false_positive_exclusions.txt` | False-positive exclusions | `sherlock_project/sites.py` |
| `.../sherlock/master/.../data.schema.json` | Remote schema conformance test | `tests/conftest.py` |
| `.../sherlock-project/sherlock/pulls/{n}` | `--json <PR number>` manifest loading | `sherlock_project/sherlock.py` |

Pass `--local` to use only the bundled manifest and avoid the first two.

## Naming

This repository has not been renamed away from the upstream package name
(`sherlock-project`) or console script (`sherlock`). It is **not published**
under that name on PyPI, and no claim is made to it. Installing
`sherlock-project` from PyPI installs upstream's release, not this work. See
[docs/README.md](docs/README.md) for how to install from this source tree.
