"""Asked when a username already has stored results, before scanning it again.

This replaced a permanent `re-scan ‹ off ›` toggle, and the trade is the point:
the toggle sat on screen for every scan and mattered on almost none of them,
while the one moment the answer matters is the moment you press SCAN on a name
already in the database. So the question moved to there, and the control went
away.

It also fixes something the toggle never could. Resume behaviour used to be
invisible until after the run: a scan of a fully-stored username skipped all 680
sites, said so only in the activity log, and otherwise looked exactly like a scan
that had found nothing. Stating it before anything starts is what makes it
legible.

**Three options, not two.** "Cancel or re-scan" leaves out the one that is
usually right. A manifest that has grown since the last run leaves a handful of
genuinely new sites, and checking only those is both the cheapest and the most
useful thing to do -- so it is offered first, with its real count.

**The counts are the content.** "This username has been scanned before, continue?"
is a question nobody can answer. "680 stored, 39 not yet checked" is.
"""

from __future__ import annotations

from typing import ClassVar, Literal

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Grid, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Static

from sherlock_project.tui.runner import ScanPlan
from sherlock_project.tui.theme import count_of

# What the caller does next. `view` exists for the case where resuming would do
# nothing at all -- offering "Resume" when there is nothing left to check is
# offering a button that does not do what its label says.
ResumeChoice = Literal["resume", "fresh", "view", "cancel"]


class ResumeScreen(ModalScreen[ResumeChoice]):
    """Resume, start over, or back out -- with the numbers for each."""

    BINDINGS: ClassVar = [
        Binding("escape", "cancel", "cancel"),
    ]

    def __init__(self, username: str, plan: ScanPlan) -> None:
        super().__init__()
        self._username = username
        self._plan = plan

    @property
    def _nothing_left(self) -> bool:
        return self._plan.to_scan == 0

    def compose(self) -> ComposeResult:
        plan = self._plan
        yield from ()
        with Vertical(id="dialog"):
            yield Label(f"{self._username} has been scanned before",
                        classes="dialog-title")

            detail = Text()
            detail.append(
                f"{count_of(plan.stored, 'result')} already stored", style="bold"
            )
            detail.append(f" of {plan.total} sites in the current list.\n")
            if self._nothing_left:
                detail.append(
                    "Nothing is left to check — every site in the list has an "
                    "answer for this username.",
                    style="dim",
                )
            else:
                detail.append(
                    f"{count_of(plan.to_scan, 'site')} not yet checked, "
                    f"usually because the site list has grown since the last "
                    f"scan.",
                    style="dim",
                )
            yield Static(detail, id="resume-detail")

            with Grid(id="resume-buttons"):
                yield Static()
                if self._nothing_left:
                    # Resuming would check zero sites, so the useful action is
                    # to go and look at what is already there.
                    yield Button("View results", variant="primary", id="resume-view")
                else:
                    yield Button(
                        f"Check {plan.to_scan} new",
                        variant="primary",
                        id="resume-continue",
                    )
                yield Button(f"Re-scan all {plan.total}", id="resume-fresh")
                yield Button("Cancel", id="resume-cancel")
            yield Label("esc cancels", classes="dim")

    def on_mount(self) -> None:
        # The non-destructive, cheapest action holds focus. Re-scanning all of
        # them is minutes of work and re-fetches evidence that already exists,
        # so it should never be the thing Enter does by accident.
        default = "#resume-view" if self._nothing_left else "#resume-continue"
        self.query_one(default, Button).focus()

    @on(Button.Pressed, "#resume-continue")
    def _continue(self) -> None:
        self.dismiss("resume")

    @on(Button.Pressed, "#resume-view")
    def _view(self) -> None:
        self.dismiss("view")

    @on(Button.Pressed, "#resume-fresh")
    def _fresh(self) -> None:
        self.dismiss("fresh")

    @on(Button.Pressed, "#resume-cancel")
    def _cancel(self) -> None:
        self.dismiss("cancel")

    def action_cancel(self) -> None:
        self.dismiss("cancel")
