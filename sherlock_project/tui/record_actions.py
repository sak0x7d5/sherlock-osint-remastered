"""The menu behind `⋯ Actions`: everything that can be done to one stored record.

It replaced a permanent `✕ Delete this username` button in the record header.
That button was the most saturated thing on the results screen, sat on every
section including the two where nobody is thinking about deleting anything, and
was the first Tab stop after the username list -- so the one irreversible action
in the app was also the easiest control to reach by accident.

Here Delete is still one step away for anyone looking for it, but it is last,
set apart by a rule, drawn red, and ends in an ellipsis because it opens a
confirmation rather than acting. Everything above it is safe.

**Each row names its key.** The menu is how the shortcuts are learned, not a
substitute for them: someone who opens it twice to export will see `^e` beside
the word and stop opening it.
"""

from __future__ import annotations

from typing import ClassVar

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Label, OptionList
from textual.widgets.option_list import Option

from sherlock_project.tui.theme import count_of

# Width of the label column, so the keys line up down the right edge.
LABEL_WIDTH = 26


def _row(label: str, key: str = "", *, style: str = "") -> Text:
    line = Text(f"{label:<{LABEL_WIDTH}}", style=style)
    line.append(f"{key:>4}", style="dim" if not style else style)
    return line


class RecordActionsScreen(ModalScreen[str | None]):
    """Pick one action for a record. Dismisses with its id, or None."""

    BINDINGS: ClassVar = [Binding("escape", "close", "close")]

    def __init__(self, username: str, *, links: int = 0) -> None:
        super().__init__()
        self._username = username
        self._links = links

    def compose(self) -> ComposeResult:
        with Vertical(id="actions-menu"):
            # Text, not markup: the username is user data.
            yield Label(Text(self._username, style="bold"), id="actions-title")
            yield OptionList(
                Option(_row("Export JSON…", "^e"), id="export"),
                Option(_row("Scan this username again…"), id="rescan"),
                Option(
                    _row(
                        f"Copy {count_of(self._links, 'found link')}"
                        if self._links
                        else "Copy found links"
                    ),
                    id="copy",
                    # Nothing to copy is a disabled row rather than a missing
                    # one, so the menu keeps the same shape for every record.
                    disabled=not self._links,
                ),
                Option(_row("Show the AI profile"), id="profile"),
                None,
                Option(
                    _row(f"Delete {self._username}…", "del", style="bold red"),
                    id="delete",
                ),
                id="actions-list",
            )

    def on_mount(self) -> None:
        # The first row, which is safe. A menu that opened on its last row
        # would put Delete under Enter.
        self.query_one(OptionList).highlighted = 0

    @on(OptionList.OptionSelected)
    def _chosen(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(event.option.id)

    def on_click(self, event) -> None:
        # A click outside the menu closes it, like any popup menu.
        if event.widget is self:
            self.dismiss(None)

    def action_close(self) -> None:
        self.dismiss(None)
