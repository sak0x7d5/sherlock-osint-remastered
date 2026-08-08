<!-- This README should be a mini version at all times for use on pypi -->

<p align=center>
  <strong><span>Hunt down social media accounts by username across 400+ social networks, with local-model profile analysis</span></strong>
  <br><br>
  <span>Full documentation is in the <a href="https://github.com/sak0x7d5/sherlock-osint-remastered">GitHub repository</a></span>
  <br>
</p>

> **Independent derivative.** This is not the [Sherlock Project](https://github.com/sherlock-project/sherlock)
> and is not affiliated with or endorsed by it. It is not published to PyPI —
> `pip install sherlock-project` installs upstream's release, not this work.
> See `NOTICE.md` in the repository for provenance.

## Usage

```console
$ sherlock --help
usage: sherlock [-h] [--version] [--verbose] [--folderoutput FOLDEROUTPUT]
                [--output OUTPUT] [--csv] [--xlsx] [--site SITE_NAME]
                [--proxy PROXY_URL] [--dump-response] [--json JSON_FILE]
                [--timeout TIMEOUT] [--print-all] [--print-found] [--no-color]
                [--browse] [--local] [--nsfw] [--txt] [--ignore-exclusions]
                [--ai] [--ai-synthesize-only] [--anchor [TRUST:]FIELD=VALUE]
                [--force-ai-synthesis]
                USERNAMES [USERNAMES ...]
```

To search for only one user:
```bash
$ sherlock user123
```

To search for more than one user:
```bash
$ sherlock user1 user2 user3
```

To extract structured profile data with a configured local model:
```bash
$ sherlock setup ai
$ sherlock --ai user123
```
