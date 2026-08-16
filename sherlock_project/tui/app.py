"""`sherlock ui` -- the three panes, and the shell that holds them.

**Three tabs, and the fourth one was cut deliberately.** Scan, Results,
Settings. The obvious fourth is an AI/model screen, and it does not exist
because it would duplicate rows the settings pane already owns: the model
picker, the endpoint, the context window and the temperature are all settings,
and putting them on their own screen too would give the same value two homes and
two ways to disagree. `setup ai` remains as a command for the guided first run;
the tab would only be a second front on the same config.

Nothing is a tab that is not a *place*. Tabs are for parallel destinations
someone moves between, so the modal things -- editing one field, picking a model
-- are dialogs pushed over whatever is underneath, because they are steps within
a task rather than places to be.

**Tab order is the order of a session, not alphabetical.** You scan, then you
read what came back, and you visit settings when something needs changing.
Opening on Scan means the first screen is the one with the only control anyone
needs on a first run.

The identity strip is docked above the tabs rather than living on one of them.
Which database is being written to does not change per tab, and an operator
console that loses that line when the tab changes is how the wrong database gets
written to without anyone noticing.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import Any, ClassVar

from rich.console import Console
from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Input, Static, TabbedContent, TabPane

from sherlock_project.ai_config import ai_config_path, try_load_settings
from sherlock_project.database import default_database_path
from sherlock_project.settings import field_values
from sherlock_project.tui.results_pane import ResultsPane
from sherlock_project.tui.scan_pane import ScanPane
from sherlock_project.tui.settings_pane import SettingsPane
from sherlock_project.tui.theme import SHERLOCK_CSS

SCAN_TAB = "tab-scan"
RESULTS_TAB = "tab-results"
SETTINGS_TAB = "tab-settings"


class SherlockUI(App[None]):
    """The unified operator screen."""

    CSS = SHERLOCK_CSS
    TITLE = "sherlock"

    # Alt+digit for the tabs, alt+q to leave. Three constraints pick these, and
    # between them they rule out most of the keyboard:
    #
    # - Bare digits are unusable: the scan pane has a text field, so "1" would
    #   change tab instead of typing a username containing a digit.
    # - Most ctrl combinations are already spoken for. `Input` alone claims
    #   ctrl+a/c/d/e/k/u/v/w/x and the ctrl+arrows, `RichLog` takes
    #   ctrl+pageup/pagedown, and `DataTable` takes ctrl+home/end -- all three
    #   are on screen here.
    # - ctrl+q and ctrl+s are XON/XOFF. On a terminal with flow control still
    #   enabled they are eaten by the tty before the app is ever told, and the
    #   symptom is a key that silently does nothing. (`sherlock settings` has
    #   used ctrl+s to save since before this screen existed; it is left alone
    #   rather than changed underneath people, but it carries the same caveat.)
    # - Function keys were the first attempt and are worse than they look: on
    #   laptops most of them need Fn held as well, and terminal emulators and
    #   desktops intercept F1 for help before the application sees it.
    #
    # `alt+*` is very nearly untouched -- `alt+backspace` is the only claim any
    # widget here makes on it -- and alt+digit for tabs is what browsers,
    # terminals and editors already do.
    BINDINGS: ClassVar = [
        Binding("alt+1", "show_tab('tab-scan')", "scan"),
        Binding("alt+2", "show_tab('tab-results')", "results"),
        Binding("alt+3", "show_tab('tab-settings')", "settings"),
        Binding("alt+q", "quit", "quit"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._settings_values = field_values(try_load_settings())

    def compose(self) -> ComposeResult:
        yield Static(id="appbar")
        with TabbedContent(initial=SCAN_TAB):
            with TabPane("SCAN", id=SCAN_TAB):
                yield ScanPane(self._settings_values)
            with TabPane("RESULTS", id=RESULTS_TAB):
                yield ResultsPane()
            with TabPane("SETTINGS", id=SETTINGS_TAB):
                yield SettingsPane()
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#appbar", Static).update(
            Text.assemble(
                ("SHERLOCK", "bold cyan"),
                ("   db ", "dim"),
                (str(default_database_path()), "dim"),
            )
        )
        # Focus the username field, so the app opens ready to be typed into.
        # It is the only control anyone needs on a first run, and a screen that
        # opens with focus somewhere unhelpful makes the first keystroke a
        # guess. Set here rather than by the pane itself for the reason
        # SettingsPane.on_mount records -- panes that grab focus on mount fight
        # each other and drag the tab bar around.
        self.query_one("#target-input", Input).focus()

    def action_show_tab(self, tab: str) -> None:
        self.query_one(TabbedContent).active = tab

    @on(TabbedContent.TabActivated)
    def _tab_activated(self, event: TabbedContent.TabActivated) -> None:
        """Give the new pane focus, and re-read anything it shows from disk.

        Without the focus half, a tab switch leaves focus on the tab bar, so the
        arrow keys move between tabs instead of down the settings rows -- the
        pane looks active and ignores every key, which reads as the app having
        hung.

        The reload half is why RESULTS is worth a visit at all. Its list is
        loaded on mount, and every pane mounts at app start, so without this the
        username you just finished scanning is missing from the tab whose entire
        job is showing what has been scanned -- and the only way to see it was a
        manual refresh nobody would know to press.
        """
        if event.pane.id == RESULTS_TAB:
            self.query_one(ResultsPane).action_reload()

        for widget in event.pane.query("*"):
            if widget.focusable:
                widget.focus()
                return

    @on(ScanPane.ShowStored)
    def _show_stored(self, event: ScanPane.ShowStored) -> None:
        """"Nothing left to check" — so go and look at what is there.

        Offered instead of a Resume button that would check zero sites. The
        scan pane raises it rather than switching tabs itself: which tab holds
        the results is this app's business, not that pane's.
        """
        self.action_show_tab(RESULTS_TAB)
        self.query_one(ResultsPane).select_username(event.username)

    @on(ResultsPane.ScanWithAnalysis)
    def _scan_with_analysis(self, event: ResultsPane.ScanWithAnalysis) -> None:
        """A profile was asked for, but nothing was extracted to build it from.

        Pass 2 merges stored Pass 1 extractions; a username scanned without
        analysis has none. So the results pane points here instead of offering
        a build that could only produce an empty profile, and this sets the
        scan up rather than making the user retype it.
        """
        self.action_show_tab(SCAN_TAB)
        self.query_one(ScanPane).prepare_analysis_scan(event.username)

    @on(SettingsPane.Closed)
    def _settings_closed(self, event: SettingsPane.Closed) -> None:
        """Escape in settings leaves the tab; it does not leave the app.

        The standalone `sherlock settings` exits on this message because exiting
        is what "done" means there. Here the same keystroke means "back to
        work", so it re-reads the settings the scan pane runs on -- a
        concurrency change that only took effect after a restart would be a
        silent lie about what the next scan will do.
        """
        self._settings_values.update(field_values(try_load_settings()))
        scan = self.query_one(ScanPane)
        scan.refresh_settings(self._settings_values)
        self.action_show_tab(SCAN_TAB)


def _can_draw(interactive: bool | None = None) -> bool:
    """Whether there is a terminal worth drawing on.

    stdout because this draws, and a redirected stdout is the case where drawing
    is wrong; stdin because a full-screen app with no key source would simply
    hang. The same test `sherlock settings` makes, for the same reason -- a
    full-screen app that takes over a CI log is worse than the crash it
    replaces.
    """
    if interactive is not None:
        return interactive
    return sys.stdout.isatty() and sys.stdin.isatty()


async def run_ui(
    argv: Sequence[str] | None = None,
    *,
    interactive: bool | None = None,
    console: Console | None = None,
) -> int:
    """Open the unified UI, or explain why it cannot open."""
    output = console or Console(highlight=False)
    if argv:
        # No flags yet, and silently ignoring one would be worse than saying so:
        # someone typing `sherlock ui --fresh` has a expectation about that run
        # that this cannot meet.
        output.print(
            Text(
                f"sherlock ui takes no arguments (got {' '.join(argv)}).",
                style="yellow",
            )
        )
        return 2

    if not _can_draw(interactive):
        output.print(
            Text(
                "No terminal to draw on, so the UI was not opened.\n"
                "Use the command line instead: sherlock <username>",
                style="dim",
            )
        )
        return 0

    # run_async, never run(). App.run() calls asyncio.run() internally and this
    # is reached from inside main()'s loop, so the synchronous form dies with
    # "asyncio.run() cannot be called from a running event loop".
    await SherlockUI().run_async()
    return 0


def config_summary() -> dict[str, Any]:
    """The stored settings, for anything that needs them without a screen."""
    return {
        "config_path": str(ai_config_path()),
        "database_path": str(default_database_path()),
        "values": field_values(try_load_settings()),
    }
