"""A yes/no dialog for actions that cannot be taken back.

Small on purpose. The only thing it adds over a keypress is a moment in which
the user reads what is about to happen -- so the caller passes the specifics
(what, how much, and that it is permanent) and this draws them.

**Defaults to cancel.** Enter on this screen does nothing destructive; deleting
takes a deliberate second keystroke on a differently-labelled control. A dialog
whose default action is the irreversible one is a dialog that gets dismissed
into data loss by muscle memory.
"""

from __future__ import annotations

from typing import ClassVar

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Grid, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Static


class ConfirmScreen(ModalScreen[bool]):
    """Ask before doing something irreversible. Dismisses True only on yes."""

    BINDINGS: ClassVar = [
        Binding("escape", "cancel", "cancel"),
    ]

    def __init__(
        self,
        title: str,
        detail: str,
        *,
        confirm_label: str = "Delete",
    ) -> None:
        super().__init__()
        self._title = title
        self._detail = detail
        self._confirm_label = confirm_label

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label(self._title, classes="dialog-title")
            yield Static(Text(self._detail), id="confirm-detail")
            with Grid(id="confirm-buttons"):
                # A spacer, so the two buttons sit together at the right rather
                # than at opposite edges of the dialog -- a pair of choices has
                # to read as a pair, and the eye should not have to travel the
                # width of the box to find the second one.
                yield Static()
                # Cancel first, and focused: the destructive control should not
                # be the one under the finger when the dialog opens.
                yield Button("Cancel", id="confirm-no")
                yield Button(self._confirm_label, variant="error", id="confirm-yes")
            yield Label("esc cancels", classes="dim")

    def on_mount(self) -> None:
        self.query_one("#confirm-no", Button).focus()

    @on(Button.Pressed, "#confirm-yes")
    def _yes(self) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#confirm-no")
    def _no(self) -> None:
        self.dismiss(False)

    def action_cancel(self) -> None:
        self.dismiss(False)
