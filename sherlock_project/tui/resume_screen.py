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

**Fetching is not the only work.** A run has two halves -- fetch pages, then
analyse them -- and only the first needs the network. So "nothing left to check"
is about sites, never about the run: a fully fetched username whose pages have
never been read still has an entire AI pass waiting, over evidence already on
disk. Offering only `Re-scan all` there sends someone through 680 refetches to
reach work that needs none, and `View results` returns them to the pane that
sent them. Both were reachable; neither did what was wanted.
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

    def __init__(
        self, username: str, plan: ScanPlan, *, analysis_on: bool = False
    ) -> None:
        super().__init__()
        self._username = username
        self._plan = plan
        self._analysis_on = analysis_on

    @property
    def _nothing_to_fetch(self) -> bool:
        return self._plan.to_scan == 0

    @property
    def _can_analyse(self) -> bool:
        """Whether resuming would run the model over pages already stored.

        Both halves are required. Stored pages with analysis off is not an
        offer -- the run would skip them exactly as the last one did -- and
        analysis on with nothing pending would promise a pass with no input.
        """
        return self._analysis_on and self._plan.pending_analysis > 0

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
            if not self._nothing_to_fetch:
                detail.append(
                    f"{count_of(plan.to_scan, 'site')} not yet checked, "
                    f"usually because the site list has grown since the last "
                    f"scan.",
                    style="dim",
                )
            elif self._can_analyse:
                # The sentence that used to be wrong. Every site has an answer,
                # and there is still a full pass of work to do -- so say which
                # work, and say that it is free.
                detail.append(
                    f"Every site has an answer, but "
                    f"{count_of(plan.pending_analysis, 'stored page')} "
                    f"{'has' if plan.pending_analysis == 1 else 'have'} not "
                    f"been analysed. Analysing them re-fetches nothing.",
                    style="dim",
                )
            else:
                detail.append(
                    "Nothing is left to check — every site in the list has an "
                    "answer for this username.",
                    style="dim",
                )
            yield Static(detail, id="resume-detail")

            buttons = Grid(id="resume-buttons")
            if self._can_analyse:
                buttons.add_class("-analysing")
            with buttons:
                # The spacer only exists to right-align the row. An analysis
                # pass takes that cell rather than adding a fifth, and the
                # modifier class re-sizes the columns for four real buttons --
                # the detail line above carries the noun, so the label only has
                # to carry the verb and the count.
                if self._can_analyse:
                    yield Button(
                        f"Analyse {plan.pending_analysis}",
                        variant="primary",
                        id="resume-analyse",
                    )
                else:
                    yield Static()
                if not self._nothing_to_fetch:
                    yield Button(
                        f"Check {plan.to_scan} new",
                        variant="primary",
                        id="resume-continue",
                    )
                else:
                    # Nothing to fetch, so the scanning button would check zero
                    # sites. Looking at what is stored is the honest offer --
                    # primary only when it is the best one available.
                    yield Button(
                        "View results",
                        variant="default" if self._can_analyse else "primary",
                        id="resume-view",
                    )
                yield Button(f"Re-scan all {plan.total}", id="resume-fresh")
                yield Button("Cancel", id="resume-cancel")
            yield Label("esc cancels", classes="dim")

    def on_mount(self) -> None:
        # The non-destructive, cheapest action holds focus. Re-scanning all of
        # them is minutes of work and re-fetches evidence that already exists,
        # so it should never be the thing Enter does by accident.
        #
        # Analysis outranks viewing when both are available: it is the work the
        # user came to do, and it still touches no network.
        if self._can_analyse:
            default = "#resume-analyse"
        elif self._nothing_to_fetch:
            default = "#resume-view"
        else:
            default = "#resume-continue"
        self.query_one(default, Button).focus()

    @on(Button.Pressed, "#resume-continue")
    def _continue(self) -> None:
        self.dismiss("resume")

    @on(Button.Pressed, "#resume-analyse")
    def _analyse(self) -> None:
        # The same choice as resuming, because it is the same run. A resumed
        # scan with nothing to fetch builds no engine and opens no connection;
        # what is left of it is the AI pass over stored pages. Naming it
        # separately here is the point -- the behaviour already existed and
        # nothing on screen had ever said so.
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
