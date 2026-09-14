"""The dialog that offers a newer release and then installs it.

Two states in one screen rather than two screens, because they are one
sentence: "there is a newer version" and "here is it arriving". Pushing a
second dialog over the first to show progress would put the thing being
described behind the thing describing it.

**Nothing is installed without a press.** The check runs on its own, but it
only ever gets as far as this dialog -- which opens with the non-committal
button focused, the same rule the delete confirmation follows. An app that
replaces itself while someone is reading about it is not offering a choice.

**The install narrates itself in the shared vocabulary.** A `phase_line`, the
same one the scan screen uses for a model load and the results pane uses for a
profile rebuild, so a slow step here looks like a slow step anywhere else. It
has to: resolving and building this project's dependency set is minutes, not
seconds, and a dialog that sits still for minutes is one somebody kills.
"""

from __future__ import annotations

from time import perf_counter
from typing import ClassVar

from rich.text import Text
from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Grid, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Static

from sherlock_project.tui.theme import REDRAW_INTERVAL, phase_line
from sherlock_project.updater import (
    InstallMode,
    Release,
    install_command,
    manual_instruction,
    run_install,
    updater_available,
)


class UpdateScreen(ModalScreen[bool]):
    """Offer a release, install it on request, and say what happened.

    Dismisses True only when an install finished cleanly, which is the caller's
    cue to show the restart note. Every other ending -- declined, failed,
    impossible for this install mode -- dismisses False, because none of them
    changes what is on disk.
    """

    BINDINGS: ClassVar = [
        Binding("escape", "cancel", "cancel"),
    ]

    def __init__(
        self,
        release: Release,
        installed: str,
        mode: InstallMode,
    ) -> None:
        super().__init__()
        self._release = release
        self._installed = installed
        self._mode = mode
        self._command = install_command(mode, release.tag)
        # Live state for the install, read by the timer ten times a second.
        self._running = False
        self._started = 0.0
        self._tick = 0
        self._state = "waiting"
        self._last_line = ""

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label("Update available", classes="dialog-title")
            yield Static(self._offer_text(), id="update-detail")
            # The status line is the working state. It replaces the buttons
            # rather than sitting beside them, so a second press cannot start a
            # competing install -- the same reason the profile pane hides its
            # controls while a rebuild runs.
            yield Static(id="update-status")
            with Grid(id="update-buttons"):
                yield Static()
                # Not now holds focus. Installing is the committing action and
                # it rewrites the program on disk; it should never be the thing
                # Enter does to a dialog someone has not finished reading.
                yield Button("Not now", id="update-no")
                yield Button("Update", variant="primary", id="update-yes")
            yield Label("esc closes", classes="dim")

    def on_mount(self) -> None:
        self.query_one("#update-status", Static).display = False
        self.query_one("#update-no", Button).focus()
        # One timer for the screen, costing an attribute check per tick while
        # nothing is installing.
        self.set_interval(REDRAW_INTERVAL, self._tick_status)

        # An install mode that cannot update itself says so up front rather
        # than behind a button that fails. The offer is still worth showing:
        # knowing a release exists is the useful half, and the instruction
        # names the command that does work.
        if self._command is None or not updater_available(self._mode):
            self.query_one("#update-yes", Button).display = False
            self.query_one("#update-no", Button).label = "Close"

    def _offer_text(self) -> Text:
        detail = Text()
        detail.append("installed  ", style="dim")
        detail.append(f"{self._installed}\n")
        detail.append("available  ", style="dim")
        detail.append(f"{self._release.version}\n", style="bold")
        detail.append(f"{self._release.url}\n", style="dim")
        if self._command is None:
            detail.append("\n")
            detail.append(
                manual_instruction(self._mode, self._release.tag), style="yellow"
            )
        elif not updater_available(self._mode):
            detail.append("\n")
            detail.append(
                f"{self._mode} is not on PATH, so this cannot install itself. "
                f"Run it yourself:\n"
                f"{manual_instruction('pipx', self._release.tag)}",
                style="yellow",
            )
        return detail

    # -- the working state --------------------------------------------------

    def _tick_status(self) -> None:
        """Draw the install as it goes, ten times a second.

        Rewritten whole each tick rather than appended to, so the line becomes
        `ready` in place instead of leaving a history of itself behind. It is
        two short lines; rebuilding them costs nothing.
        """
        if not self._running:
            return
        self._tick += 1
        elapsed = perf_counter() - self._started

        text = Text()
        text.append_text(
            phase_line("update", self._state, elapsed=elapsed, tick=self._tick)
        )
        if self._last_line:
            text.append("\n")
            # The installer's own last word, dimmed. Resolving dependencies
            # prints steadily for minutes, and one moving line is the
            # difference between "working" and "hung".
            text.append(self._last_line[:70], style="dim")
        self.query_one("#update-status", Static).update(text)

    @work(exclusive=True, group="update-install")
    async def _install(self) -> None:
        assert self._command is not None
        code, last = await run_install(self._command, on_line=self._note_line)

        self._running = False
        status = self.query_one("#update-status", Static)
        if code == 0:
            elapsed = perf_counter() - self._started
            status.update(
                phase_line("update", "ready", elapsed=elapsed, tick=self._tick)
            )
            self.dismiss(True)
            return

        # Failure stays on screen, with the command to run by hand. A failure
        # that disappears after five seconds is a failure nobody can act on,
        # and this one has a real chance of being a locked file on Windows --
        # where the files being replaced belong to the process replacing them.
        failed = Text()
        failed.append_text(
            phase_line(
                "update", "failed",
                elapsed=perf_counter() - self._started, tick=self._tick,
            )
        )
        failed.append(f"\nexit {code}: {last}\n", style="red")
        failed.append("Run it yourself:\n", style="dim")
        failed.append(manual_instruction(self._mode, self._release.tag))
        status.update(failed)
        self.query_one("#update-buttons").display = True
        self.query_one("#update-yes", Button).display = False
        self.query_one("#update-no", Button).label = "Close"
        self.query_one("#update-no", Button).focus()

    def _note_line(self, line: str) -> None:
        """Keep the installer's latest output for the next repaint.

        Stored rather than drawn, for the reason the scan feed exists: pip
        prints far faster than a screen can usefully repaint, and touching a
        widget per line would spend the install animating instead of
        installing.
        """
        self._last_line = line
        if self._state == "loading" and "Installing" in line:
            self._state = "installing"

    # -- buttons ------------------------------------------------------------

    @on(Button.Pressed, "#update-yes")
    def _yes(self) -> None:
        if self._command is None:
            return
        # The whole container goes, never the individual cells: Textual's grid
        # skips hidden children, so hiding one button slides its neighbour into
        # the wrong column and silently clips the label.
        self.query_one("#update-buttons").display = False
        self.query_one("#update-status", Static).display = True
        self._running = True
        self._started = perf_counter()
        self._state = "loading"
        self._last_line = ""
        self._tick_status()
        self._install()

    @on(Button.Pressed, "#update-no")
    def _no(self) -> None:
        self.dismiss(False)

    def action_cancel(self) -> None:
        # Escape during an install closes the dialog, not the install. The
        # subprocess is already replacing files on disk and killing it halfway
        # is the one outcome worse than waiting for it.
        self.dismiss(False)
