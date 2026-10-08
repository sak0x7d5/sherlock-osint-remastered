"""`sherlock-rm ui` -- the three panes, and the shell that holds them.

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
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.css.query import NoMatches
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
    #   symptom is a key that silently does nothing. (`sherlock-rm settings` has
    #   used ctrl+s to save since before this screen existed; it is left alone
    #   rather than changed underneath people, but it carries the same caveat.)
    # - Function keys were the first attempt and are worse than they look: on
    #   laptops most of them need Fn held as well, and terminal emulators and
    #   desktops intercept F1 for help before the application sees it.
    #
    # `alt+*` is very nearly untouched -- `alt+backspace` is the only claim any
    # widget here makes on it -- and alt+digit for tabs is what browsers,
    # terminals and editors already do.
    #
    # The tab keys are not in the footer. Each tab's label carries its digit
    # instead ("1 SCAN"), which is where someone looking for "how do I get to
    # results" is already looking -- and the footer, which ran out of width at
    # 120 columns and dropped them off the end anyway, keeps its room for the
    # keys of the screen in front of you.
    #
    # EACH TAB HAS TWO KEY NAMES, and the second is not decoration. Most
    # terminals send alt+digit as ESC followed by the digit -- GNOME Terminal,
    # Konsole and xterm on Linux, iTerm2 and Terminal.app on macOS with "Option
    # as Meta" on -- and Textual decodes ESC 1/2/3 as the characters ¡ ™ £,
    # which is what Option+1/2/3 types on a US Mac layout. So `alt+1` alone
    # only ever fired on terminals with a richer key protocol (kitty, WezTerm,
    # foot), and in the rest the tab keys silently did nothing. The character
    # names catch those terminals, and a Mac with Option left as a compose key.
    #
    # The character names are PRIORITY bindings, because the username field
    # has focus from launch and would otherwise take ¡ as text before the app
    # ever saw it. The cost is that those three characters cannot be typed into
    # it, and no site's username rules allow them anyway.
    BINDINGS: ClassVar = [
        Binding("alt+1", "show_tab('tab-scan')", "scan", show=False),
        Binding("alt+2", "show_tab('tab-results')", "results", show=False),
        Binding("alt+3", "show_tab('tab-settings')", "settings", show=False),
        Binding(
            "inverted_exclamation_mark", "show_tab('tab-scan')", "scan",
            show=False, priority=True,
        ),
        Binding(
            "trade_mark_sign", "show_tab('tab-results')", "results",
            show=False, priority=True,
        ),
        Binding(
            "pound_sign", "show_tab('tab-settings')", "settings",
            show=False, priority=True,
        ),
        Binding("alt+q", "quit", "quit"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._settings_values = field_values(try_load_settings())
        # Set once an update has actually been written to disk, and never
        # cleared: the appbar has to keep saying so for the rest of the
        # session, because the running process is still the old build and that
        # stays true until someone restarts it.
        self._update_installed = False

    def compose(self) -> ComposeResult:
        yield Static(id="appbar")
        with TabbedContent(initial=SCAN_TAB):
            # The digit is the alt+digit that opens the tab, dim so the word
            # still reads first.
            with TabPane(_tab_title(1, "SCAN"), id=SCAN_TAB):
                yield ScanPane(self._settings_values)
            with TabPane(_tab_title(2, "RESULTS"), id=RESULTS_TAB):
                yield ResultsPane()
            with TabPane(_tab_title(3, "SETTINGS"), id=SETTINGS_TAB):
                yield SettingsPane()
        yield Footer()

    def on_mount(self) -> None:
        self._redraw_appbar()
        # Focus the username field, so the app opens ready to be typed into.
        # It is the only control anyone needs on a first run, and a screen that
        # opens with focus somewhere unhelpful makes the first keystroke a
        # guess. Set here rather than by the pane itself for the reason
        # SettingsPane.on_mount records -- panes that grab focus on mount fight
        # each other and drag the tab bar around.
        self.query_one("#target-input", Input).focus()

        # Last, and only when asked for. Read straight off the flattened
        # values rather than through `runner.resolved`: that module imports the
        # Playwright engine at module scope, and a settings-only launch should
        # not pay a browser import to decide whether to make one HTTP request.
        # The value is always a real bool here -- unlike [ai], the [update]
        # section is not optional, so pydantic fills the default when a config
        # file predates it.
        if self._settings_values.get("update.check_on_startup"):
            self._check_for_update()

    def _redraw_appbar(self) -> None:
        """The identity strip, plus anything that must outlive a tab change.

        Rebuilt rather than appended to, so the update note cannot be added
        twice by two paths that both thought they were the one to add it.
        """
        # The path is cut from the MIDDLE when the bar is short of room. It
        # used to wrap instead, onto a second line this one-row bar does not
        # have -- so a long path showed as "db" and nothing at all, the one
        # thing this strip exists to say. The ends are the parts that identify
        # it: the root says whose, the file name says which.
        room = max(16, self.size.width - 20 - (52 if self._update_installed else 0))
        line = Text.assemble(
            ("SHERLOCK", "bold cyan"),
            ("   db ", "dim"),
            (_middle_truncate(str(default_database_path()), room), "dim"),
        )
        line.no_wrap = True
        line.overflow = "ellipsis"
        if self._update_installed:
            # Names a thing the reader can do from where they are sitting. A
            # note that said "re-run pip" would be pointing at a command line
            # they are not at, which is the failure `render_profile`'s
            # notes_hint already exists to avoid.
            line.append(
                "   · updated — quit with alt+q and reopen to apply",
                style="bold yellow",
            )
        self.query_one("#appbar", Static).update(line)

    @work(exclusive=True, group="update-check")
    async def _check_for_update(self) -> None:
        """Ask whether there is a newer release, and offer it if there is.

        Everything slow or fallible is inside the worker, including the
        imports: `updater` pulls in `requests` and the dialog pulls in the
        theme, and neither should be paid for by a launch with the setting off.

        Silence is the normal outcome. No release published, no network, a tag
        that will not parse, and a version this build cannot read all end here
        with nothing drawn -- an update check is not worth a dialog that says
        it failed.
        """
        from sherlock_project import __version__
        from sherlock_project.tui.update_screen import UpdateScreen
        from sherlock_project.updater import (
            checked_recently,
            detect_install,
            fetch_latest,
            is_newer,
            record_check,
        )

        if checked_recently():
            return

        release = await fetch_latest()
        if release is None:
            # Deliberately NOT stamped. Recording a failed reach would suppress
            # the next six hours of checks because the forge was briefly
            # unreachable, which is the opposite of what the stamp is for.
            return
        record_check()

        if not is_newer(release.version, __version__):
            return

        def applied(installed: bool | None) -> None:
            if installed:
                self._update_installed = True
                self._redraw_appbar()

        self.push_screen(
            UpdateScreen(release, __version__, detect_install()), applied
        )

    def on_resize(self) -> None:
        self._redraw_appbar()

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

        SCAN reloads for the same reason and it is the same class of bug. See
        `_reload_settings`: arriving at that tab is the moment its answer has to
        be current, whichever route was taken to get there.
        """
        if event.pane.id == RESULTS_TAB:
            self.query_one(ResultsPane).action_reload()
        elif event.pane.id == SCAN_TAB:
            self._reload_settings()

        if event.pane.id == RESULTS_TAB:
            # The pane knows which of its widgets is visible to take focus --
            # at narrow widths the list is a hidden picker, and the first
            # focusable widget in walk order was that hidden list.
            self.query_one(ResultsPane).focus_default()
            return
        for widget in event.pane.query("*"):
            if widget.focusable:
                widget.focus()
                return

    def _reload_settings(self) -> None:
        """Re-read the stored settings into the dict the scan pane runs on.

        HUNG ON ARRIVING AT THE SCAN TAB, not on leaving the settings one, and
        that is the fix rather than an implementation detail. The reload used to
        live on `SettingsPane.Closed`, which is posted by ESCAPE alone -- so
        changing a setting, saving it with ^S and then leaving by any other
        route (alt+1, or clicking the SCAN tab) left the pane on the values it
        was built with. Settings reach disk on save; the pane was told on close;
        the two are different keystrokes and nothing connected them.

        Not a display bug, which is how it was reported. The same dict is what
        `run_scan_session` reads its transport, concurrency, timeout and proxy
        from, and what `ai_is_configured` is asked about -- so a model chosen,
        saved, and left behind with alt+1 produced a scan that warned "no model
        is configured" and ran with analysis off while the model sat on disk.

        Reading from disk, deliberately, rather than from the settings pane's
        working copy: unsaved edits are not settings yet, and the scan must
        describe what it will actually do.

        Mutates in place. This dict is the SAME OBJECT handed to `ScanPane` at
        construction, so rebinding it here would leave the pane holding the old
        one and quietly undo the whole fix.
        """
        self._settings_values.update(field_values(try_load_settings()))
        try:
            self.query_one(ScanPane).refresh_settings(self._settings_values)
        except NoMatches:
            # The initial tab activates while its content is still mounting, so
            # the pane's own widgets are not queryable yet. It draws itself from
            # these same values in its `on_mount`, so there is nothing to do --
            # and a startup crash to refresh a screen that is about to refresh
            # itself would be a poor trade.
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

    @on(ResultsPane.Rescan)
    def _rescan(self, event: ResultsPane.Rescan) -> None:
        """Scan a stored username again: set it up on SCAN, start nothing.

        Starting the run from a menu item on another tab would be doing more
        than was asked; pressing SCAN there brings up the resume dialog with
        the real counts, which is where the choice belongs.
        """
        self.action_show_tab(SCAN_TAB)
        self.query_one(ScanPane).prepare_scan(event.username)

    @on(SettingsPane.Closed)
    def _settings_closed(self, event: SettingsPane.Closed) -> None:
        """Escape in settings leaves the tab; it does not leave the app.

        The standalone `sherlock-rm settings` exits on this message because exiting
        is what "done" means there. Here the same keystroke means "back to
        work".

        The reload is NOT done here any more. It hangs off arriving at the scan
        tab instead, which this switch triggers -- because Escape was only ever
        one of the ways out of settings, and hanging the refresh on it meant
        alt+1 and a mouse click both left the pane stale. One owner, reached by
        every route. See `_reload_settings`.
        """
        self.action_show_tab(SCAN_TAB)


def _tab_title(number: int, name: str) -> str:
    return f"[dim]{number}[/] {name}"


def _middle_truncate(text: str, width: int) -> str:
    """`/home/someone/…/sherlock/sherlock.db`: both ends kept, the middle cut."""
    if len(text) <= width:
        return text
    keep = width - 1
    head = keep // 3
    tail = keep - head
    return f"{text[:head]}…{text[-tail:]}"


def _can_draw(interactive: bool | None = None) -> bool:
    """Whether there is a terminal worth drawing on.

    stdout because this draws, and a redirected stdout is the case where drawing
    is wrong; stdin because a full-screen app with no key source would simply
    hang. The same test `sherlock-rm settings` makes, for the same reason -- a
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
        # someone typing `sherlock-rm ui --fresh` has a expectation about that run
        # that this cannot meet.
        output.print(
            Text(
                f"sherlock-rm ui takes no arguments (got {' '.join(argv)}).",
                style="yellow",
            )
        )
        return 2

    if not _can_draw(interactive):
        output.print(
            Text(
                "No terminal to draw on, so the UI was not opened.\n"
                "Use the command line instead: sherlock-rm <username>",
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
