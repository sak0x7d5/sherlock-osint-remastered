# sherlock-osint-remastered

**Find accounts by username across 719 sites — then have a local model tell you
who they belong to, with every claim traced back to the page it came from.**

![Python](https://img.shields.io/badge/python-3.13%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Regression](https://github.com/sak0x7d5/sherlock-osint-remastered/actions/workflows/regression.yml/badge.svg)

```console
$ sherlock-rm show hackerman1337
[*] Stored results for 'hackerman1337' (last scanned 2026-09-10 14:22:07 · 680 sites checked · nothing written)
[+] GitHub: https://github.com/hackerman1337
[+] Mastodon: https://mastodon.social/@hackerman1337
[+] Last.fm: https://last.fm/user/hackerman1337
[+] Bandcamp: https://bandcamp.com/hackerman1337 (no browser)
[!] 34 sites of 680 gave no answer: 21 inconclusive, 13 blocked by bot protection
    Not the same as "not found". List them: sherlock-rm show hackerman1337 --unresolved
[*] AI profile for 'hackerman1337' (built 2026-09-10 14:31:52 · extracted by Qwen3-8B-Q4_K_M)
╭─ Profile: hackerman1337 ──────────────────────────────────────────────────────╮
│ [!] No anchors used, so these values may describe different people who share  │
│ this username.                                                                │
│                                                                               │
│ CONFIDENT                                                                     │
│ display_name   Hackerman                     3 sites: GitHub, Mastodon +1     │
│ location       Berlin, DE                    2 sites: Mastodon, Last.fm       │
│ languages      Rust, Python                  2 sites: GitHub, Codeberg        │
│                                                                               │
│ UNSURE                                                                        │
│ employer       Contoso GmbH                  1 site: Mastodon                 │
│                                                                               │
│ [!] 2 notes about how this was built; run with --verbose to read them.        │
╰───────────────────────────────────────────────────────────────────────────────╯
```

<sub>Illustrative output. `--sources` replaces each `3 sites: …` summary with the
full URL behind every value.</sub>

<!-- TODO: add a screenshot of `sherlock-rm ui` here (docs/images/ui.png).
     Upstream's demo.png showed the old URL-list output, i.e. the thing this
     fork replaces, so it was removed rather than reused. -->

## Why this exists

Every tool in this category answers the same question — *does this username
exist on these sites?* — and hands you a list of URLs. The work that actually
matters starts after that list: opening thirty tabs, reading thirty profile
pages, and deciding by hand whether they describe one person.

This fork does that step for you, locally, and shows its working.

| | |
| - | - |
| **A stealth browser does the fetching** | Sites that hide behind bot protection or build their profile page in JavaScript answer a real browser. A plain HTTP request gets an empty shell and reports a live account as absent. A browser-free mode is available when speed matters more than accuracy. |
| **Every result is kept** | Results persist to SQLite, including *which transport* produced each row. Re-running resumes instead of re-scanning, and a fast run never overwrites better evidence from a browser run. |
| **A local model reads the pages** | Two passes: per-site extraction of structured facts, then cross-site synthesis into one profile where every field records which sites support it and how confident the merge is. |

Nothing leaves your machine except requests to the sites being checked, to the
model endpoint you configure yourself, and one call per scan to GitHub's API to
see whether a newer release exists. That last one is listed in
[NOTICE.md](NOTICE.md) with everything else this tool talks to; it fails
silently and never blocks a scan.

### Compared to the alternatives

Accurate as of September 2026. Where this tool is behind, the row says so.

| | [Sherlock](https://github.com/sherlock-project/sherlock) | [Maigret](https://github.com/soxoj/maigret) | [Blackbird](https://github.com/p1ngul1n0/blackbird) | **this** |
| - | - | - | - | - |
| Sites | ~400 | 3000+ | ~600 (WhatsMyName) | 719 (WhatsMyName) |
| Renders JavaScript | no | no | no | **yes** (stealth browser) |
| Reads profile page content | no | rule-based parsing | metadata extraction | **local LLM, two passes** |
| Where the analysis runs | — | your machine | hosted API, daily quota, [sends site names only](https://p1ngul1n0.gitbook.io/blackbird/ai) | **your machine, over full page text** |
| Per-claim source attribution | no | no | no | **yes** |
| Inconclusive ≠ not found | no | no | no | **yes** |
| Resumable, stored evidence | no | no | no | **yes** (SQLite) |
| Report formats | txt, csv | HTML, PDF | PDF, CSV, JSON | txt, JSON *(no PDF yet)* |
| Install | `pip` | `pip` | `pip`, clone | clone, `git+`, Docker *(no PyPI)* |

Two honest caveats about that table. Maigret checks four times as many sites,
and if raw coverage is what you need, use Maigret. And this tool has no PDF or
HTML report yet, which Maigret and Blackbird both ship.

What is genuinely different here are the five middle rows: pages are rendered
before they are read, the analysis runs on your hardware over the actual text of
each page, every value names the sites behind it, a site that could not be
determined is reported as undetermined rather than as absent, and none of it is
thrown away between runs.

> [!IMPORTANT]
> **This is not Sherlock, and it does not speak for Sherlock.**
>
> It began as a fork of the [Sherlock Project](https://github.com/sherlock-project/sherlock)
> and has since become a different tool:
>
> - **Independent.** Not affiliated with, endorsed by, or supported by the
>   Sherlock Project or its maintainers. Please do not report anything you find
>   here to them, or anything you find there to us.
> - **Not their package.** `pip install sherlock-project` installs *upstream's*
>   release. This one installs as `sherlock-rm` and does not collide with it.
> - **Overwhelmingly new code.** 86% of the Python here is in files that do not
>   exist upstream — the stealth-browser fetch engine, the SQLite persistence
>   layer, bounded content extraction, the two-pass local-model pipeline, and
>   the terminal UI. What is inherited is the site-iteration and result
>   modelling, the scan loop's argument surface, and the TXT output format.
> - **Different data, too.** Scans run on the
>   [WhatsMyName](https://github.com/WebBreacher/WhatsMyName) dataset, not
>   upstream's manifest, which is retained only for `--json`.
> - **Attribution is kept, not minimised.** Upstream's copyright stays in
>   [LICENSE](LICENSE) because the MIT licence requires it and because it is
>   accurate. [NOTICE.md](NOTICE.md) itemises exactly what is inherited, what is
>   new, what is third-party, and which upstream services this still calls.

> [!NOTE]
> **A rename is coming.** `sherlock-rm` is an interim name. Carrying "sherlock"
> invites exactly the confusion the box above spends five bullets dispelling,
> and the project has outgrown being described as a version of something else.
> A new name will land before the first tagged release; the repository will
> redirect, and the command will change with it.

## Install

Needs **Python 3.13+** (see [Requirements](#requirements) — the floor is real).
There is no published package yet, so install from the repository.
`pip install sherlock-project` installs **upstream's** release, not this work.

**One line, no clone.** This puts a `sherlock-rm` command on your PATH, which is
what every example below assumes. The name is deliberate: it does not collide
with upstream's `sherlock`, so you can keep both installed.

```bash
pipx install git+https://github.com/sak0x7d5/sherlock-osint-remastered
# or: uv tool install git+https://github.com/sak0x7d5/sherlock-osint-remastered
sherlock-rm --version
```

**Docker**, if you would rather not have Python 3.13 on the host — which is the
practical answer on Debian 12, Ubuntu 24.04, and RHEL 9:

```bash
git clone https://github.com/sak0x7d5/sherlock-osint-remastered
cd sherlock-osint-remastered
docker build -t sherlock-osint-remastered .
docker run --rm sherlock-osint-remastered someusername
```

The stealth browser binary is not baked into the image; the first browser-backed
run downloads it. Mount a volume at the browser cache and at
`/root/.local/share/sherlock` to keep both across containers.

**From a clone**, with a plain virtualenv:

```bash
git clone https://github.com/sak0x7d5/sherlock-osint-remastered
cd sherlock-osint-remastered
python3.13 -m venv .venv && source .venv/bin/activate
pip install .
```

**To work on it.** Poetry adds the dev dependencies and the test suite:

```bash
poetry install
poetry run sherlock-rm --help
tox
```

Installed this way `sherlock` is not on your PATH, so every `sherlock ...`
below becomes `poetry run sherlock ...` (or run `poetry shell` once).

## Requirements

- **Python 3.13 or newer.** This floor is real, not tidiness — the scan loop
  uses an `asyncio` feature added in 3.13 and the package will not install on
  3.10–3.12. If your distribution ships an older Python, use the Docker image
  above rather than fighting it.
- [llama.cpp](https://github.com/ggml-org/llama.cpp)'s `llama-server` on your
  PATH, plus some `.gguf` models, for the optional AI passes. Sherlock runs the
  server for you — you never start or stop one:

  ```
  sherlock-rm setup ai
  ```

  It looks where models usually already are (LM Studio, the llama.cpp and
  HuggingFace caches, `./models`). If yours live somewhere else, say so once
  with `--models-dir <folder>`; **any layout works**, it searches inside.

  Every model found becomes selectable, and switching costs no restart — they
  load on demand, so expect the first request after a switch to take tens of
  seconds.

  Already running your own `llama-server`? It is used exactly as it is and
  never restarted or stopped — Sherlock only manages a server it started
  itself. That covers single-model servers (`-m model.gguf`) too.

  Other OpenAI-compatible runtimes (Ollama, vLLM, plain endpoints) are not
  supported.
- The first browser-backed run downloads a stealth Chromium build.

### Which model

**[Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B-GGUF) at `Q4_K_M`** is the
recommended starting point: roughly 5 GB on disk and about 6 GB of VRAM or RAM
in use, which fits an 8 GB card or a 16 GB laptop without swapping. It is the
smallest thing tested here that reliably returns well-formed structured fields
in Pass 1.

Download a `Q4_K_M` build from [Qwen/Qwen3-8B-GGUF](https://huggingface.co/Qwen/Qwen3-8B-GGUF)
into wherever you keep `.gguf` files, then point Sherlock at that folder once:

```bash
sherlock-rm setup ai --models-dir <folder>
```

You do not start `llama-server` yourself — Sherlock starts and stops the one it
uses.

Smaller models will run, and mostly cost you Pass 1 precision — they invent
fields or file facts under the wrong heading. Larger models help most in Pass 2,
where the merge decisions live. Whatever you pick is recorded against every
extraction, and `sherlock-rm show` names it, so a profile built by a model you no
longer trust is identifiable rather than silently mixed in.

## Quick start

Scan a username:

```bash
sherlock-rm someusername
```

Scan several at once:

```bash
sherlock-rm user1 user2 user3
```

Scan, then have the local model read what it found:

```bash
sherlock-rm setup ai
sherlock-rm --ai someusername
```

Look at what is already stored, without scanning or writing anything:

```bash
sherlock-rm show someusername
```

Trade accuracy for speed by skipping the browser entirely:

```bash
sherlock-rm --no-webbrowser someusername
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

`sherlock-rm show <user> --sources` prints the full URL behind every value.
`--json` emits the whole thing machine-readably.

### Anchoring

By default a synthesis merges every account sharing the username **without
deciding they are the same person**, and the report says so. If you already know
something true about your subject, pass it as an anchor and accounts that
contradict it are weighted accordingly:

```bash
sherlock-rm --ai --anchor "name=Jane Doe" --anchor "verified:location=Berlin" janedoe
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
inconclusive rather than a guess. `sherlock-rm show <user> --unresolved` lists
those sites. Sites blocked by bot protection or rejecting the username's format
land here too, and they are *not* the same as sites where the username was free.

**The model can be wrong.** Pass 2 output is an inference over scraped pages,
not a record. Treat it as a lead to verify, and use `--sources` to check what
each claim actually rests on.

### How long a scan takes

<!-- TODO: fill these from a real run before publishing. Suggested method:
     `time sherlock-rm --fresh <user>` and `time sherlock-rm --fresh --no-webbrowser <user>`
     on a stated connection and machine, best of three, 680 sites, --concurrency 30.
     Blackbird publishes 731 sites in 44s for comparison, so leaving this blank
     reads worse than a slow honest number. -->

| Default sites | Transport | Wall clock |
| - | - | - |
| 680 | stealth browser | *not yet published* |
| 680 | `--no-webbrowser` | *not yet published* |

The browser path is the slower one by a wide margin — it starts Chromium and
renders each page — and that cost is the whole reason `--no-webbrowser` exists.
The AI passes are separate from both and are bounded by your model and hardware,
not by the network.

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

**Removing someone.** On the RESULTS tab of `sherlock-rm ui`, hover a username and
press the `✕` at the end of its row — or select it and press `delete`. Either
way it names what will go and asks first, then erases that username's results,
stored page content, extractions and profile. Nothing else is touched.

> [!NOTE]
> **There is no command-line equivalent yet.** Outside the UI, removing someone
> still means deleting the database file or opening it with an external SQLite
> tool. For scripted runs, pointing `SHERLOCK_DB` at a scratch file is the way
> to keep a scan from being retained at all.

## Sites

719 sites across 20 categories, from the
[WhatsMyName](https://github.com/WebBreacher/WhatsMyName) dataset, vendored
unmodified and separately licensed (CC BY-SA 4.0 — see [NOTICE.md](NOTICE.md)).
The dataset ships 720 entries; one (`7cup`) is flagged `valid: false` upstream
and is skipped at load, so 719 are usable.

| Category | Sites | | Category | Sites |
| - | - | - | - | - |
| social | 204 | | shopping | 23 |
| gaming | 74 | | music | 22 |
| tech | 57 | | blog | 18 |
| hobby | 50 | | art | 12 |
| coding | 47 | | dating | 12 |
| NSFW | 39 | | political | 11 |
| misc | 35 | | health | 11 |
| finance | 26 | | news | 10 |
| business | 25 | | video | 9 |
| images | 25 | | archived | 9 |

The 39 NSFW sites are skipped unless you pass `--nsfw`, leaving **680** checked
by default. No network call is needed to load the list; it ships with the
package.

This is a curated list, not the largest one available — Maigret carries roughly
four times as many. The trade is deliberate: every site here has a two-sided
rule, so it can report *undetermined* instead of guessing, and every hit is a
page worth handing to the model.

[docs/sites.md](docs/sites.md) lists the 478 sites in the legacy upstream
manifest, which is reachable only via `--json` and is not what a normal scan
uses.

## Commands

```
sherlock-rm USERNAME...        Scan for a username
sherlock-rm show USERNAME      Read what is stored. Never scans, never writes
sherlock-rm ui                 Full-screen interface: scan, results and settings
sherlock-rm setup ai           Configure the local model endpoint
sherlock-rm settings           Edit stored defaults for scans, output and AI
```

`sherlock-rm ui` puts all three in one place and needs a terminal; without one it
says so and exits rather than failing.

> [!NOTE]
> **On macOS, turn on Option-as-Meta first.** The UI drives everything from
> `alt` — `alt+1/2/3` for the tabs, `alt+q` to quit, `alt+a` and `alt+f` in the
> scan pane. macOS terminals send composed characters instead by default, so
> `alt+3` types `#` into the username field rather than switching tab.
>
> - **Terminal.app** — Settings → Profiles → Keyboard → *Use Option as Meta key*
> - **iTerm2** — Settings → Profiles → Keys → *Left Option key* → `Esc+`
>
> Without it the tabs are still clickable, and `Tab` eventually reaches the tab
> bar where `←`/`→` switch tabs — but the shortcuts in the footer will not work.

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
<summary><strong><code>sherlock-rm show</code> options</strong></summary>

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

Bug reports, false positives, false negatives, and site requests each have an
[issue template](https://github.com/sak0x7d5/sherlock-osint-remastered/issues/new/choose);
a false negative with the site name and the transport used is the single most
useful thing you can file.

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
