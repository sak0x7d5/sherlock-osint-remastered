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

from rich.console import RenderableType
from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
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
        detail: str | RenderableType,
        *,
        confirm_label: str = "Delete",
        cancel_label: str = "Cancel",
        danger: bool = False,
    ) -> None:
        super().__init__(classes="-danger" if danger else None)
        self._title = title
        # A plain string is wrapped in Text so user data in it -- a username
        # with brackets -- is never read as markup. A renderable is the
        # caller's own styling and is drawn as given.
        self._detail = Text(detail) if isinstance(detail, str) else detail
        self._confirm_label = confirm_label
        self._cancel_label = cancel_label

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            # Text, so a username with brackets in it is not read as markup;
            # a Static, so a long one wraps inside the dialog instead of
            # running off its edge. The title carries the name now that the
            # button does not.
            yield Static(Text(self._title), classes="dialog-title")
            yield Static(self._detail, id="confirm-detail")
            # Right-aligned together, so a pair of choices reads as a pair, and
            # sized to their labels -- the confirm label names what it deletes,
            # and a fixed grid column clipped a long username off the end.
            with Horizontal(id="confirm-buttons"):
                # Cancel first, and focused: the destructive control should not
                # be the one under the finger when the dialog opens.
                yield Button(self._cancel_label, id="confirm-no")
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
