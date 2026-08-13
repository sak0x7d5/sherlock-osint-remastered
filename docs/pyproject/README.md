<!-- This README should be a mini version at all times for use on pypi -->

<p align=center>
  <strong><span>Find accounts by username across 720 sites, then analyse them with a local model</span></strong>
  <br><br>
  <span>Full documentation is in the <a href="https://github.com/sak0x7d5/sherlock-osint-remastered">GitHub repository</a></span>
  <br>
</p>

> **Independent derivative.** This is not the [Sherlock Project](https://github.com/sherlock-project/sherlock)
> and is not affiliated with or endorsed by it. It is not published to PyPI —
> `pip install sherlock-project` installs upstream's release, not this work.
> See `NOTICE.md` in the repository for provenance.

Requires Python 3.13 or newer.

## Usage

```console
$ sherlock --help
usage: sherlock [-h] [--version] [--verbose] [--folderoutput FOLDEROUTPUT]
                [--output OUTPUT] [--site SITE_NAME] [--proxy PROXY_URL]
                [--dump-response] [--json JSON_FILE] [--timeout TIMEOUT]
                [--concurrency COUNT] [--print-all] [--print-found]
                [--no-color] [--browse] [--local] [--nsfw]
                [--no-webbrowser | --webbrowser] [--txt]
                [--ignore-exclusions] [--fresh] [--ai]
                [--ai-synthesize-only] [--anchor [TRUST:]FIELD=VALUE]
                [--force-ai-synthesis]
                USERNAMES [USERNAMES ...]
```

Scan one or more usernames:
```bash
$ sherlock user123
$ sherlock user1 user2 user3
```

Extract structured profile data with a configured local model, then read the
result back without scanning again:
```bash
$ sherlock setup ai
$ sherlock --ai user123
$ sherlock show user123
```

Skip the browser for a faster, less accurate scan:
```bash
$ sherlock --no-webbrowser user123
```

## Responsible use

This assembles publicly visible information about people into a profile. Have a
lawful reason, treat the local database as personal data you are responsible
for, and do not republish captured page content. See the repository README for
the full statement.
