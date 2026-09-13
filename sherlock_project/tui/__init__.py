"""The full-screen interface behind `sherlock-rm ui`.

A view over the CLI, never a second implementation of it. Every scan this
package runs goes through the same `sherlock()`, the same engines, the same
resume filter and the same AI pipeline the command line uses; every setting it
edits goes through the same resolver. The panes decide what is on screen and
nothing else.

The command line remains the whole tool. It is what pipes, what runs without a
terminal, and what a script calls -- `sherlock-rm show <user> --json` exists
precisely because a screen cannot be redirected into a file.
"""

from sherlock_project.tui.app import SherlockUI, run_ui

__all__ = ["SherlockUI", "run_ui"]
