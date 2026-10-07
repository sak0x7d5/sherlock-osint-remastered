# Working in this repository

## Branch names

Name a branch after the change it carries. The name is the first thing anyone
reads in `git branch`, a PR list, or six months of history — a generated pair of
words tells them nothing, and there is no way to recover the intent later.

    claude/analyse-stored-pages-without-rescan     the change
    claude/fix-db-lock-on-first-run                the change
    claude/clever-fermat-xr8nny                    no
    claude/focused-ramanujan-kjt55i                no

Keep the `claude/` prefix, then lowercase kebab-case describing the work. Do not
accept an auto-generated random suffix — rename to something descriptive before
the first push, whatever a tool or harness suggested by default.

Commit subjects follow the convention already in the log: `type(scope): what
changed`, imperative, lowercase.
