<p align="center">
  <br>
  <img src="images/sherlock-logo.png" alt="sherlock"/>
  <br>
  <span>Hunt down social media accounts by username across 400+ social networks</span>
  <br>
</p>

<p align="center">
  <a href="#installation">Installation</a>
  &nbsp;&nbsp;&nbsp;•&nbsp;&nbsp;&nbsp;
  <a href="#general-usage">Usage</a>
  &nbsp;&nbsp;&nbsp;•&nbsp;&nbsp;&nbsp;
  <a href="../NOTICE.md">Provenance</a>
</p>

<p align="center">
<img width="70%" height="70%" src="images/demo.png" alt="demo"/>
</p>

> [!IMPORTANT]
> This is an **independent derivative** of the [Sherlock Project](https://github.com/sherlock-project/sherlock),
> not affiliated with or endorsed by it. It is not the upstream project and is
> not the `sherlock-project` package on PyPI. See [NOTICE.md](../NOTICE.md) for
> what is inherited, what is new, and which upstream services it still calls.

## What this adds

On top of upstream's username enumeration, this derivative fetches profiles
with a stealth browser, persists every result to SQLite, and runs an optional
two-pass analysis against a **local** language model: per-site structured
extraction, then cross-site profile synthesis with per-field provenance. No
profile content leaves the machine except to the sites being checked and to
the local model endpoint you configure.

## Installation

Install from this source tree. There is no published package for this
derivative — `pipx install sherlock-project` installs *upstream's* release,
not this one.

```bash
git clone https://github.com/sak0x7d5/sherlock-osint-remastered
cd sherlock-osint-remastered
poetry install
poetry run sherlock --help
```

Requires Python 3.13 or newer. The first browser-backed run downloads a
stealth Chromium binary.

To use the AI passes, configure a local model endpoint once:

```bash
poetry run sherlock setup ai
```

## General usage

To search for only one user:
```bash
sherlock user123
```

To search for more than one user:
```bash
sherlock user1 user2 user3
```

Accounts found will be stored in an individual text file with the corresponding username (e.g ```user123.txt```).

```console
$ sherlock --help
usage: sherlock [-h] [--version] [--verbose] [--folderoutput FOLDEROUTPUT] [--output OUTPUT] [--csv] [--xlsx] [--site SITE_NAME] [--proxy PROXY_URL] [--dump-response]
                [--json JSON_FILE] [--timeout TIMEOUT] [--print-all] [--print-found] [--no-color] [--browse] [--local] [--nsfw] [--txt] [--ignore-exclusions]
                USERNAMES [USERNAMES ...]

Sherlock: Find Usernames Across Social Networks (Version 0.16.0)

positional arguments:
  USERNAMES             One or more usernames to check with social networks. Check similar usernames using {?} (replace to '_', '-', '.').

options:
  -h, --help            show this help message and exit
  --version             Display version information and dependencies.
  --verbose, -v, -d, --debug
                        Display extra debugging information and metrics.
  --folderoutput FOLDEROUTPUT, -fo FOLDEROUTPUT
                        If using multiple usernames, the output of the results will be saved to this folder.
  --output OUTPUT, -o OUTPUT
                        If using single username, the output of the result will be saved to this file.
  --csv                 Create Comma-Separated Values (CSV) File.
  --xlsx                Create the standard file for the modern Microsoft Excel spreadsheet (xlsx).
  --site SITE_NAME      Limit analysis to just the listed sites. Add multiple options to specify more than one site.
  --proxy PROXY_URL, -p PROXY_URL
                        Make requests over a proxy. e.g. socks5://127.0.0.1:1080
  --dump-response       Dump the HTTP response to stdout for targeted debugging.
  --json JSON_FILE, -j JSON_FILE
                        Load data from a JSON file or an online, valid, JSON file. Upstream PR numbers also accepted.
  --timeout TIMEOUT     Time (in seconds) to wait for response to requests (Default: 60)
  --print-all           Output sites where the username was not found.
  --print-found         Output sites where the username was found (also if exported as file).
  --no-color            Don't color terminal output
  --browse, -b          Browse to all results on default browser.
  --local, -l           Force the use of the local data.json file.
  --nsfw                Include checking of NSFW sites from default list.
  --txt                 Enable creation of a txt file
  --ignore-exclusions   Ignore upstream exclusions (may return more false positives)
```

## Credits

This work stands on the [Sherlock Project](https://github.com/sherlock-project/sherlock)
and everyone who has [contributed to it](https://github.com/sherlock-project/sherlock/graphs/contributors). ❤️

<a href="https://github.com/sherlock-project/sherlock/graphs/contributors">
  <img src="https://contrib.rocks/image?&columns=25&max=10000&&repo=sherlock-project/sherlock" alt="upstream contributors"/>
</a>

## License

MIT.

- © 2019 Sherlock Project — upstream work this derivative is built on.
- © 2026 sherlock-osint-remastered contributors — changes made here.

See [LICENSE](../LICENSE) for the full text and [NOTICE.md](../NOTICE.md) for
provenance.
