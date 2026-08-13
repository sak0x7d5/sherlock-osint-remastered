<!--
  PLACEHOLDER — PROJECT LOGO
  This project currently has no mark of its own. Upstream Sherlock's detective
  logo lives at docs/images/sherlock-logo.png and is deliberately NOT used here:
  it is upstream's visual identity, MIT covers their code rather than their
  branding, and reusing it works against a fork that is trying to be legible as
  a separate thing. Drop a distinct logo in docs/images/ and reference it above
  the title when one exists.
-->

# sherlock-osint-remastered

**Find accounts by username across 720 sites — then have a local model tell you
who they belong to, with every claim traced back to the page it came from.**

![Python](https://img.shields.io/badge/python-3.13%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Regression](https://github.com/sak0x7d5/sherlock-osint-remastered/actions/workflows/regression.yml/badge.svg)

> [!IMPORTANT]
> This is an **independent derivative** of the [Sherlock Project](https://github.com/sherlock-project/sherlock).
> It is not affiliated with, endorsed by, or supported by it, and it is **not**
> the `sherlock-project` package on PyPI. Do not report issues found here to
> upstream. See [NOTICE.md](NOTICE.md) for what is inherited, what is new, and
> which upstream services this still calls.

<!--
  PLACEHOLDER — DEMO SCREENSHOTS
  Two images belong here and neither exists yet:
    1. a scan in progress (the rich terminal reporter)
    2. the pass-two profile table, which is this fork's real differentiator
  docs/images/demo.png is upstream's asset showing upstream's plain URL-list
  output. It is not used here because it no longer resembles what this tool
  prints, and advertising a fork with the parent's screenshot is misleading.

  When capturing replacements, two hard rules:
    - Use a FICTIONAL persona. A pass-two screenshot displays an inferred
      identity plus the sites it was inferred from; that is a dossier, not a
      URL list. persona-replacements.txt (kept outside this repo, beside
      CLAUDE.md) already fixes a persona used by the prompt examples — reuse it
      rather than inventing a second.
    - Capture against a throwaway database (`SHERLOCK_DB=/tmp/demo.db`).
      Clearing the terminal is not enough; the real database carries the scan
      the screenshot came from.
-->

## Why this exists

Upstream answers one question well: *does this username exist on these sites?*
The answer is a list of URLs, and reading it is left to you.

This fork carries that work forward and adds the step after it: taking the pages
behind those URLs and turning them into a picture of who the account holder
appears to be — locally, on your own hardware, with the evidence attached.

Three things are different here:

| | |
| - | - |
| **A stealth browser does the fetching** | Sites that hide behind bot protection or build their profile page in JavaScript answer a real browser. A plain HTTP request gets an empty shell and reports a live account as absent. A browser-free mode is available when speed matters more than accuracy. |
| **Every result is kept** | Results persist to SQLite, including *which transport* produced each row. Re-running resumes instead of re-scanning, and a fast run never overwrites better evidence from a browser run. |
| **A local model reads the pages** | Two passes: per-site extraction of structured facts, then cross-site synthesis into one profile where every field records which sites support it and how confident the merge is. |

Nothing leaves your machine except requests to the sites being checked and to
the model endpoint you configure yourself.

## Requirements

- **Python 3.13 or newer.** This floor is real, not tidiness — the scan loop
  uses an `asyncio` feature added in 3.13 and the package will not install on
  3.10–3.12.
- A local model server for the optional AI passes. Currently
  [LM Studio](https://lmstudio.ai) — `sherlock setup ai` auto-detects it.
- The first browser-backed run downloads a stealth Chromium build.

## Install

There is no published package. `pip install sherlock-project` installs
**upstream's** release, not this work.

```bash
git clone https://github.com/sak0x7d5/sherlock-osint-remastered
cd sherlock-osint-remastered
poetry install
poetry run sherlock --help
```

## Quick start

Scan a username:

```bash
sherlock someusername
```

Scan several at once:

```bash
sherlock user1 user2 user3
```

Scan, then have the local model read what it found:

```bash
sherlock setup ai
sherlock --ai someusername
```

Look at what is already stored, without scanning or writing anything:

```bash
sherlock show someusername
```

Trade accuracy for speed by skipping the browser entirely:

```bash
sherlock --no-webbrowser someusername
```

## How the analysis works

The AI layer runs in two passes, both against your local endpoint.

**Pass 1 — per-site extraction.** Each page that produced a hit is reduced to
bounded metadata plus its main content, then handed to the model, which returns
structured fields. Stored per site, and cached: re-running does not re-extract
unless the prompt contract changed or you pass `--fresh`.

**Pass 2 — cross-site synthesis.** Every stored extraction for a username is
merged into one profile. Fields that several sites agree on are marked
confident; fields resting on weaker or conflicting evidence are separated out
rather than blended in. Each value carries the sites that support it.

`sherlock show <user> --sources` prints the full URL behind every value.
`--json` emits the whole thing machine-readably.

### Anchoring

By default a synthesis merges every account sharing the username **without
deciding they are the same person**, and the report says so. If you already know
something true about your subject, pass it as an anchor and accounts that
contradict it are weighted accordingly:

```bash
sherlock --ai --anchor "name=Jane Doe" --anchor "verified:location=Berlin" janedoe
```

## Accuracy, honestly

**Two fetch engines, and they do not agree.** The browser engine is the default
and the accurate one. `--no-webbrowser` uses plain HTTPS requests: it never
starts Chromium, which is the actual win, but it runs no JavaScript. A
client-rendered profile arrives as an empty shell, so **a real account can be
reported as absent.** This was measured against Instagram, not theorised.

Every stored row records the transport that produced it, so the damage is
contained: a later browser scan re-checks anything the fast path could not
confirm, and a fast "not found" never gets mistaken for a settled answer.

**UNKNOWN is a real answer.** The site rules are two-sided — one pattern proves
presence, another proves absence. When neither matches, the result is
inconclusive rather than a guess. `sherlock show <user> --unresolved` lists
those sites. Sites blocked by bot protection or rejecting the username's format
land here too, and they are *not* the same as sites where the username was free.

**The model can be wrong.** Pass 2 output is an inference over scraped pages,
not a record. Treat it as a lead to verify, and use `--sources` to check what
each claim actually rests on.

## Responsible use

This tool collects publicly visible information about people and assembles it
into a profile. That is a meaningfully different act from checking whether a
username is taken, and it carries obligations.

- **Have a reason.** Authorised security assessments, investigating your own
  accounts and exposure, research with an ethics framework behind it. Not
  stalking, harassment, doxxing, or building files on people who have not
  consented.
- **Public does not mean unrestricted.** Aggregating scattered public facts into
  one profile can be regulated even when every individual fact is public.
  Depending on where you and your subject are, that may engage GDPR or
  equivalent law — including obligations you cannot satisfy after the fact.
- **You are the data controller.** The database on your disk holds personal
  data about real people. Where it lives, who can read it, how long you keep it,
  and deleting it when you are done are your responsibility.
- **Scraping may breach a site's terms of service** regardless of whether the
  data is public. That is between you and the site.
- **Do not republish captured page content.** Extractions and stored responses
  are working data, not something to commit to a repository or attach to a
  report.

The authors provide this under the MIT licence, without warranty, and are not
responsible for how you use it.

## Where your data lives

Nothing is written to the directory you run from. State follows the user:

| | Windows | Linux | macOS |
| - | - | - | - |
| Database | `%LOCALAPPDATA%\sherlock\sherlock.db` | `~/.local/share/sherlock/sherlock.db` | `~/Library/Application Support/sherlock/sherlock.db` |
| Settings | `%LOCALAPPDATA%\sherlock\config.toml` | `~/.config/sherlock/config.toml` | `~/Library/Application Support/sherlock/config.toml` |

Override with `SHERLOCK_DB` and `SHERLOCK_CONFIG`. Pointing `SHERLOCK_DB` at a
scratch file gives you a throwaway database — the right move for demos, tests,
and anything you would rather not keep.

The database stores captured page content, extracted fields, and synthesised
profiles, keyed by username. It persists until you delete it.

> [!NOTE]
> **There is currently no built-in way to delete a subject from the database.**
> Removing someone means deleting the database file or editing it with an
> external SQLite tool. A `sherlock forget <username>` command is the obvious
> gap; see TODO.

## Sites

720 sites across 20 categories, from the
[WhatsMyName](https://github.com/WebBreacher/WhatsMyName) dataset, vendored
unmodified and separately licensed (CC BY-SA 4.0 — see [NOTICE.md](NOTICE.md)).

| Category | Sites | | Category | Sites |
| - | - | - | - | - |
| social | 205 | | shopping | 23 |
| gaming | 74 | | music | 22 |
| tech | 57 | | blog | 18 |
| hobby | 50 | | art | 12 |
| coding | 47 | | dating | 12 |
| NSFW | 39 | | political | 11 |
| misc | 35 | | health | 11 |
| finance | 26 | | news | 10 |
| business | 25 | | video | 9 |
| images | 25 | | archived | 9 |

The 39 NSFW sites are skipped unless you pass `--nsfw`, leaving **681** checked
by default. No network call is needed to load the list; it ships with the
package.

<!--
  PLACEHOLDER — docs/sites.md IS STALE
  It lists 478 sites and says it was generated by devel/site-list.py, which
  reads the legacy upstream data.json rather than the WMN manifest that is
  actually shipped. Regenerate it against wmn-data.json, or delete it and let
  this section stand alone. Not linked from here until it is one or the other.
-->

## Commands

```
sherlock USERNAME...        Scan for a username
sherlock show USERNAME      Read what is stored. Never scans, never writes
sherlock setup ai           Configure the local model endpoint
sherlock settings           Edit stored defaults for scans, output and AI
```

<details>
<summary><strong>Scan options</strong></summary>

| Option | Effect |
| - | - |
| `--ai` | Extract structured profile data with the configured local model |
| `--ai-synthesize-only` | Build profiles from existing extractions without scanning |
| `--anchor [TRUST:]FIELD=VALUE` | Add a run-only identity anchor. Repeatable |
| `--force-ai-synthesis` | Rebuild profiles even when inputs are unchanged |
| `--fresh` | Re-scan every site instead of resuming. With `--ai`, also re-extracts — this is how a newly configured model reaches results you already have |
| `--webbrowser` / `--no-webbrowser` | Force the stealth browser on or off for one run |
| `-c`, `--concurrency COUNT` | Sites checked at once (default 30) |
| `--site SITE_NAME` | Limit to named sites. Repeatable |
| `--timeout SECONDS` | Per-request timeout (default 60) |
| `--nsfw` | Include the 39 NSFW sites |
| `--proxy URL` | Route requests through a proxy, e.g. `socks5://127.0.0.1:1080` |
| `--local` | Guarantee the bundled site list. The default already uses it and makes no network call, so this is belt-and-braces |
| `--json FILE` | Load a different site manifest: file, URL, or upstream PR number |
| `--ignore-exclusions` | Legacy only. Upstream's false-positive exclusion list applies to a legacy manifest loaded via `--json`; on the default path nothing is fetched and this does nothing |
| `--output`, `--folderoutput`, `--txt` | Write results to a file |
| `--print-all` / `--print-found` | Include sites where the username was absent / only where found |
| `--browse` | Open every result in your browser |
| `--dump-response` | Dump raw HTTP responses for debugging |
| `--verbose`, `-v` | Diagnostics and metrics |
| `--no-color` | Plain output |

</details>

<details>
<summary><strong><code>sherlock show</code> options</strong></summary>

| Option | Effect |
| - | - |
| `--accounts` | Only the accounts found |
| `--profile` | Only the synthesised profile |
| `--unresolved` | Sites that gave no answer — inconclusive, blocked, or rejecting the username format |
| `--sources` | Full URL of every site backing each value, instead of a count |
| `--json` | Machine-readable output |
| `--verbose`, `-v` | Include diagnostic notes recorded during synthesis |

</details>

## Development

```bash
tox                  # lint + tests with coverage — the push gate
tox -e offline       # same tests, no coverage, faster
tox -e lint          # ruff only
tox -e online        # live-site probes; network dependent, never a gate
```

<!--
  PLACEHOLDER — CONTRIBUTING
  No CONTRIBUTING.md exists. Before inviting contributions, decide: are pull
  requests wanted at all, and if so what is expected of them (tests, commit
  message shape, whether new sites belong here or upstream in WhatsMyName).
  Issue templates already exist under .github/ISSUE_TEMPLATE/.
-->

## Credits

Built on the [Sherlock Project](https://github.com/sherlock-project/sherlock)
and everyone who [contributed to it](https://github.com/sherlock-project/sherlock/graphs/contributors).

Site data from [WhatsMyName](https://github.com/WebBreacher/WhatsMyName) by
Micah Hoffman and contributors.

## License

MIT.

- © 2019 Sherlock Project — the upstream work this is built on
- © 2026 sherlock-osint-remastered contributors — changes made here

See [LICENSE](LICENSE) for the full text and [NOTICE.md](NOTICE.md) for
provenance.
