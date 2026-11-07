# Provenance

`sift-osint` is an **independent derivative work** of the
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

That file carries two copyright lines: the 2019 line is the upstream Sherlock
Project's, from which this work derives, and the 2026 line covers changes made
here. The licence body itself is left verbatim -- explanatory prose inside it
stops GitHub identifying the file as MIT, which is why that explanation lives
in this file instead.

## Inherited from upstream

- The legacy site manifest (`sift_osint/resources/data.json`) and its
  schema. Retained and still shipped, but no longer the manifest a scan uses by
  default — see "Third-party data: WhatsMyName" below.
- Site iteration and result modelling (`sites.py`, `result.py`).
- The scan loop's argument surface and the TXT output format.
- Issue templates and the regression workflow. Upstream's exclusions updater and
  site-list workflows were inherited and have since been removed.

## Third-party data: WhatsMyName

The site manifest is being migrated to the WhatsMyName (WMN) dataset, which is
**separately licensed** from this repository's code.

| | |
| - | - |
| Project | WhatsMyName |
| Source | <https://github.com/WebBreacher/WhatsMyName> |
| License | [CC BY-SA 4.0](http://creativecommons.org/licenses/by-sa/4.0/) |
| Copyright | © Micah Hoffman |
| Contributors | 3xp0rt, AlexisPradoOficial, funnyzak, mbiesiad, Micah Hoffman, mtedholm |
| Vendored at | `sift_osint/resources/wmn-data.json` |

The dataset is vendored **unmodified**, with its embedded license block intact,
and is kept as a standalone file rather than merged into any other manifest.
ShareAlike applies to the data and works derived from it; it does not reach this
repository's MIT-licensed code, which merely reads the file. Adaptation to the
internal site-record shape happens at load time in `wmn_adapter.py` and is not
written back to disk, so the distributed copy stays byte-identical to upstream's.

Refreshing the dataset is a straight download from the source above — no
re-derivation step — which keeps that property easy to verify.

## Added in this repository

None of the following exists upstream:

- Stealth-browser fetching via Playwright (`playwright_engine.py`).
- SQLite result persistence with schema migration (`database.py`).
- Bounded profile-content preparation (`content_extraction.py`).
- Two-pass local-model analysis: per-site structured extraction plus
  cross-site profile synthesis (`ai_engine.py`, `profile_synthesis.py`,
  `synthesis_pipeline.py`, `investigation_context.py`).
- A local AI provider integration and its setup flow (`ai_provider.py`,
  `ai_config.py`, `ai_setup.py`), plus an opt-in hosted provider (Gemini) and
  its client-side pacing (`ai_rate_limit.py`).
- The `rich`-based terminal reporter (`notify.py`).

## Services still consumed from upstream

These are deliberate runtime dependencies on upstream-operated resources, not
oversights. They are disclosed here so that operators know what this tool
talks to:

| Resource | Used for | Defined in |
| - | - | - |
| `https://data.sherlockproject.xyz` | Live site manifest | `sift_osint/sites.py` |
| `.../sherlock/refs/heads/exclusions/false_positive_exclusions.txt` | False-positive exclusions | `sift_osint/sites.py` |
| `.../sherlock/master/.../data.schema.json` | Remote schema conformance test | `tests/conftest.py` |
| `.../sherlock-project/sherlock/pulls/{n}` | `--json <PR number>` manifest loading | `sift_osint/sherlock.py` |

Pass `--local` to use only the bundled manifest and avoid the first two.

Further outbound calls are not upstream's but are listed here for the same
reason, since the table above is what an operator reads to learn what this
tool talks to:

| Resource | Used for | Defined in |
| - | - | - |
| `https://api.github.com/repos/sak0x7d5/sift-osint/releases/latest` | Update check, once per scan, and once when the UI opens if `update.check_on_startup` is on | `sift_osint/__init__.py`, `sift_osint/updater.py` |
| `https://github.com/sak0x7d5/sift-osint` | Downloading a release, and only after the update dialog is confirmed | `sift_osint/updater.py` |
| `https://generativelanguage.googleapis.com/v1beta/openai` | Both AI passes, **only** when `ai.provider = "gemini"` is configured; sends the text of every profile page found | `sift_osint/ai_provider.py` |

The check runs off-thread with a 10s timeout, and any failure -- including the
404 returned while no release exists -- is swallowed into a verbose-only debug
line, so it can neither block nor fail a scan.

The startup check is **off by default** and is the only one of these that is
configurable: `update.check_on_startup` in the config file, or the Update
section of `sift-osint settings`. It is rate-limited to one call every six
hours by a stamp in the user cache directory (`SIFT_CACHE` overrides the
location). The download happens only when someone presses Update in the dialog
that the check raises, so the second row is reached by an explicit action
rather than by the check itself.

## Naming

This project is named `sift-osint` (distribution, import package `sift_osint`,
and console script alike). It was previously called `sherlock-rm`, an interim
name that carried the upstream project's; that name was never released.

It makes no claim to upstream's names. `sherlock-project` on PyPI is upstream's
release, not this work, and the two can be installed side by side. The one
inherited name left is the module `sift_osint/sherlock.py` (the scan loop),
kept because it is inherited code; see "Inherited from upstream" above.

For anyone who ran it under the old name: the old per-user data directory
(`sherlock`) and `SHERLOCK_*` environment variables are still honoured as a
fallback, so no database or settings are lost. See
[README.md](README.md#where-your-data-lives).
