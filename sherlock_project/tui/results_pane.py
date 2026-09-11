"""The results pane: pick a username on the left, read what is stored on the right.

Master/detail, because the question is always "this one, tell me more" and a
single flat list cannot answer it without either truncating the detail or
hiding the names.

**The detail is three switched sections, not one long scroll.** This started as
a single stacked column on the argument that accounts, unresolved sites and the
profile are read together. Stacking them was wrong for a reason that only shows
up with real data: a `DataTable` is itself a scrollable viewport, so a table
inside a scrolling column produces TWO vertical scrollbars side by side, plus a
horizontal one when the URLs are long. Forty accounts made the pane look broken,
and it also meant scrolling past all forty to reach the profile.

Exactly one scroll region is visible at a time now. The switcher is a `Tabs`
strip driving a `ContentSwitcher` rather than a nested `TabbedContent`, because
a second full tab bar directly under the first reads as two competing
navigations; a compact strip reads as what it is, a section switch inside one
place.

**The counts live on the tab labels** -- `UNRESOLVED 6`, not `UNRESOLVED`. That
is what preserves the property the stacked layout was protecting: an account
list means something different once you know six sites gave no answer, and now
you can see that without leaving the accounts. Hiding the number was the only
real cost of switching, so it is the one thing that does not get hidden.

**A symbol column comes with a key, where the symbols contend.** The mark
column is two cells wide and its header is blank, which left the distinction
this tool exists to make -- blocked is not inconclusive is not rejected --
drawn in symbols nobody had been shown a glossary for. The live feed on the
scan pane does not have that problem because it prints the word beside every
symbol; a table has no room for that, so the sentence goes above the column
instead, naming the symbols that section is actually drawing.

It sits on its own line rather than beside the tabs, which is where the spare
width looks like it is. Measured, that space is not spare: `Tabs` is a
scrolling strip, so squeezed it drops tabs rather than wrapping or
ellipsizing them, silently and from the left. A key sharing that row needs a
126-column terminal to leave three tabs intact and 138 for four -- below that
it eats the way back to ACCOUNTS. One row of table is the cheaper thing to
spend, and it is only spent on UNRESOLVED: ACCOUNTS is one symbol repeated
down a list the tab already names, and PROFILE has no symbols at all.

**Everything that is not interactive is still a Rich renderable inside a
`Static`.** Widgets earn their keep when something can be clicked or focused --
the accounts and unresolved lists are, so they are tables; the profile is not,
so it is drawn.

**Viewing never writes; every write is an explicit, named action.** That is the
guarantee `show.py` exists to make, restated for a pane that has since grown
two of them -- deleting a username, and building a profile. The rule it is
protecting is not "no writes" but the one that made `--ai-synthesize-only`
unsuitable as a viewer: LOOKING at a profile must not be able to destroy one.
So selection, section switching and scrolling touch nothing, deletion asks
first, a first build passes `force=False` because there is nothing to replace,
and a rebuild passes `force=True` because that is what the button says.

**A rebuild keeps the profile's own anchors.** `--ai-synthesize-only` takes them
from the command line and never from the profile it overwrites, so rebuilding
without re-typing `--anchor` abandons the identity resolution rather than
recomputing it -- measured on real data as 4 fields / 7 values / 2 anchors
becoming 11 fields / 73 values / 0. Here they are seeded from the stored profile
and shown above the button, so the safe default is also the visible one.
"""

from __future__ import annotations

import webbrowser
from io import StringIO
from pathlib import Path
from time import perf_counter
from typing import Any, ClassVar

from rich.console import Console, Group
from rich.text import Text
from textual import events, on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Grid, Vertical, VerticalScroll
from textual.coordinate import Coordinate
from textual.message import Message
from textual.widgets import (
    Button,
    ContentSwitcher,
    DataTable,
    Static,
    Tab,
    Tabs,
)
from textual.widgets.data_table import CellDoesNotExist

# Section ids. The tab and the pane it selects need SEPARATE ids even though
# they are the same section: two widgets sharing one id makes `query_one`
# ambiguous, and which of the two it returns depends on walk order rather than
# on what the caller asked for. `ContentSwitcher` selects a child by id and
# `Tabs` reports the activated tab by id, so the mapping below is what connects
# them -- one table, rather than a string transformation at each call site.
SEC_ACCOUNTS = "sec-accounts"
SEC_UNRESOLVED = "sec-unresolved"
SEC_PROFILE = "sec-profile"

TAB_ACCOUNTS = "tab-accounts"
TAB_UNRESOLVED = "tab-unresolved"
TAB_PROFILE = "tab-profile"

SECTION_FOR_TAB = {
    TAB_ACCOUNTS: SEC_ACCOUNTS,
    TAB_UNRESOLVED: SEC_UNRESOLVED,
    TAB_PROFILE: SEC_PROFILE,
}

from sherlock_project.database import (
    SherlockDB,
    StoredUsernameListing,
    default_database_path,
)
from sherlock_project.profile_synthesis import IdentityAnchor
from sherlock_project.result import QueryStatus
from sherlock_project.tui.anchor_screen import AnchorScreen
from sherlock_project.tui.confirm_screen import ConfirmScreen
from sherlock_project.tui.reporter import TuiReporter
from sherlock_project.tui.theme import (
    REDRAW_INTERVAL,
    count_of,
    elapsed_label,
    phase_line,
    spinner,
    status_from_name,
    status_key,
    status_style,
)

# Width the profile renderer is asked to lay out against. Fixed rather than the
# live pane width: `render_profile` is a three-column layout, and re-rendering
# it on every resize to chase the exact pane width costs a full re-render for a
# result nobody is measuring with a ruler. Wide enough that the columns are not
# compressed, narrow enough to fit a half-screen detail pane.
PROFILE_RENDER_WIDTH = 96

# The delete control, and the nothing a row shows the rest of the time. Three
# cells each, so the control is a BLOCK rather than a lone character: the same
# shape the `+` beside ANCHORS uses, which is the app's existing answer to "this
# symbol is a button". A bare glyph in a column of numbers reads as another
# value, and a value is not something you press.
#
# `✕` rather than a wastebasket emoji: the emoji is two cells wide in some
# terminals and one in others, so a column sized for it is wrong somewhere, and
# it falls back to a hollow box wherever the font has no colour glyph for it.
# `✕` is already this app's mark for a rejected result, so the visual vocabulary
# is not growing to pay for this either.
#
# `Text` and not a markup string, for the reason every cell here is: `str` cells
# are parsed as markup, and cell content in this pane sits next to user data.
DELETE_LABEL = " ✕ "
NO_DELETE = Text("   ")

# The stylesheet names the control's colours; this names the style. Component
# classes are how a widget exposes a part of itself to CSS, so the chip follows
# the theme like everything else instead of carrying a literal red into a file
# whose first rule is that colours are semantic tokens.
DELETE_STYLE = "username-list--delete"


class UsernameList(DataTable):
    """The stored-username picker, with a delete control on the pointed-at row.

    Erasing a username was already possible and reachable only by pressing
    `delete`. The footer named the key, but the list itself showed no sign that
    a row could be removed at all -- so the one action in the app that destroys
    a dossier was the one with no control to press, and the documented answer
    for "remove this person" was still to open the database in an external
    SQLite tool.

    **The control is drawn on the hovered row only.** A dim `✕` on every row was
    the alternative and it is worse in both directions: a column of delete marks
    reads as a list of things queued for deletion, and it parks an irreversible
    control one misclick away from every name on screen. Drawn under the
    pointer, it is exactly as present as the row it belongs to, and the row
    tint that `DataTable` already paints on hover says which row that is.

    **Clicking it does not select the row.** Cancelling the confirmation has to
    leave the pane exactly as it was, and moving the cursor on the way to a
    dialog you then dismiss is a change nobody asked for -- it would also swap
    the detail on the right for a username you decided not to touch. The base
    class moves the cursor in its own click handler, so this one runs first
    (subclass before base, in MRO order) and calls `prevent_default()`, which
    stops the dispatcher before `DataTable._on_click` is reached.

    Nothing here erases anything. The list knows which row was pointed at and
    that is all it knows; what a username costs to delete, and the asking, are
    the pane's business.
    """

    DELETE_COLUMN = "delete"

    COMPONENT_CLASSES: ClassVar[set[str]] = {DELETE_STYLE}

    class DeleteRequested(Message):
        """The ✕ on a row was pressed. Deleting is the pane's decision."""

        def __init__(self, username_list: UsernameList, username: str) -> None:
            super().__init__()
            self.username_list = username_list
            self.username = username

        @property
        def control(self) -> UsernameList:
            """The list the control was on.

            Named `control` because that is the attribute `@on(..., selector)`
            matches against -- the same contract `DataTable.RowSelected` and
            every other framework message keeps, so this one can be handled
            exactly like them.
            """
            return self.username_list

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # What the cursor is allowed to repaint. `DataTable` sends its cursor
        # colours through TWICE: once as the base style under the cell, and
        # again over the top of it -- "css" priority means that second pass
        # happens, and a cell cannot keep a colour of its own on the selected
        # row at all. The control came out white there, in the cursor's own
        # colour, on the row it is most likely to be used from; a delete button
        # painted as part of the row highlight is not a delete button.
        #
        # "renderable" drops the second pass and leaves those colours as the
        # base, which is what a base is for. A cell that states no colour still
        # takes the cursor's -- the username and the site total do, so a
        # selected row still reads as selected -- and a cell that states one
        # keeps it: the control, and the hit count, whose green is the only
        # thing on the row saying an account was found.
        #
        # Set here rather than as class attributes because these are `__init__`
        # arguments: assigned in the class body they are overwritten by the
        # defaults the moment `DataTable.__init__` runs. Both are named even
        # though the background already defaults to "renderable", so a change
        # to that default cannot silently take the control's block with it.
        self.cursor_foreground_priority = "renderable"
        self.cursor_background_priority = "renderable"
        # Which row is currently drawing its ✕, so the one before it can be
        # cleared without repainting the column on every mouse move.
        self._marked_row: int | None = None

    def _delete_chip(self) -> Text:
        """The control, in the theme's colours.

        Resolved per draw rather than built once at import: a component style
        is only knowable once the widget has an app and a theme behind it, and
        the theme can change while the app is running.
        """
        return Text(DELETE_LABEL, style=self.get_component_rich_style(DELETE_STYLE))

    def clear(self, columns: bool = False) -> UsernameList:
        # Row indices do not survive a reload: the same number is a different
        # username afterwards, or no row at all. Forgetting the mark here is
        # what stops a ✕ being left behind on a row nobody is pointing at --
        # and it means that after a delete the control has to be re-hovered,
        # which is the right amount of friction for the second one.
        self._marked_row = None
        return super().clear(columns)

    def on_mouse_move(self, event: events.MouseMove) -> None:
        self._mark(self._row_under(event.style.meta))

    def on_leave(self, _: events.Leave) -> None:
        # The pointer can leave the list without a last move across an empty
        # row, and a ✕ still drawn once it has gone is a button belonging to
        # nothing.
        self._mark(None)

    def on_click(self, event: events.Click) -> None:
        row = self._row_under(event.style.meta)
        if row is None:
            return
        cell = Coordinate(row, event.style.meta.get("column", -1))
        if not self.is_valid_coordinate(cell):
            return
        key = self.coordinate_to_cell_key(cell)
        if key.column_key.value != self.DELETE_COLUMN:
            return
        username = key.row_key.value
        if not username:
            return
        # The click was the button. Nobody else acts on it -- least of all the
        # base class, which would move the cursor onto the row being deleted.
        event.prevent_default()
        event.stop()
        self.post_message(self.DeleteRequested(self, str(username)))

    @staticmethod
    def _row_under(meta: dict[str, Any]) -> int | None:
        """Which row the pointer is on, or None if it is not on one.

        `out_of_bounds` is `DataTable`'s word for the empty space past the last
        row, which it still reports as the nearest row so a row cursor keeps its
        highlight down there. A button has to be stricter than a highlight:
        there is no row under the pointer, so there is no control to offer.
        Row -1 is the header, which is not a username either.
        """
        if not meta or meta.get("out_of_bounds"):
            return None
        row = meta.get("row")
        if not isinstance(row, int) or row < 0:
            return None
        return row

    def _mark(self, row: int | None) -> None:
        """Move the delete control onto `row`, or off the list entirely."""
        if row == self._marked_row:
            return
        for index, glyph in (
            (self._marked_row, NO_DELETE),
            (row, self._delete_chip()),
        ):
            if index is None:
                continue
            cell = Coordinate(index, 0)
            if not self.is_valid_coordinate(cell):
                continue
            try:
                self.update_cell(
                    self.coordinate_to_cell_key(cell).row_key,
                    self.DELETE_COLUMN,
                    # A copy, for the same reason the rows are built with one:
                    # the table keeps what it is given, and a renderable shared
                    # between cells is shared until something edits it.
                    glyph.copy(),
                )
            except CellDoesNotExist:
                # The column is added by whoever builds the list. Without it
                # there is no control to draw, which is not a reason to take
                # the pane down from inside a mouse handler.
                continue
        self._marked_row = row


class ResultsPane(Vertical):
    """What the database already knows, for every username in it."""

    # alt+arrows for the sections, matching alt+digit for the tabs -- one
    # modifier for "move between places", whichever level you are on. They
    # replaced `[` and `]`, which needed AltGr on several common layouts and so
    # were not the free keys they look like on a US keyboard.
    #
    # A binding at all, rather than relying on the strip's own arrow keys,
    # because reaching the strip means cycling focus with Tab: true, and
    # undiscoverable. This way the sections appear in the footer.
    BINDINGS: ClassVar = [
        Binding("ctrl+r", "reload", "refresh"),
        Binding("s", "toggle_sources", "sources"),
        Binding("v", "toggle_notes", "notes"),
        Binding("ctrl+e", "export", "export json"),
        Binding("alt+right", "next_section", "next section"),
        Binding("alt+left", "prev_section", "prev section", show=False),
        Binding("delete", "delete_username", "delete"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._listings: list[StoredUsernameListing] = []
        self._show_sources = False
        self._show_notes = False
        self._selected: str | None = None
        self._record: dict[str, Any] | None = None
        # A username to open once the list has loaded, when something sent us
        # here to look at one in particular.
        self._pending_selection: str | None = None
        # Anchors for a profile built from this pane. Run-only, like the scan
        # pane's -- and edited with the SAME dialog, not a second one.
        self._build_anchors: list[IdentityAnchor] = []
        # Which username `_build_anchors` belong to, so switching rows reseeds
        # them from that profile rather than carrying one person's anchors onto
        # another.
        self._anchors_for: str | None = None
        # Row key -> the URL that row is about, so Enter can open it.
        self._account_urls: dict[str, str] = {}
        # Which statuses each section is currently DRAWING, so the key above
        # the table names those and only those. Filled by the fill methods
        # rather than listed here as what each section may contain: a key
        # offering `rejected` when nothing was rejected is noise in a line
        # whose whole job is to be short enough to read in passing. A section
        # that registers nothing gets no key, which is how ACCOUNTS and
        # PROFILE opt out -- keyed by section rather than hardcoded to the one
        # that uses it, so a section added later only has to say what it draws.
        self._section_statuses: dict[str, list[QueryStatus]] = {}
        # Live state for a rebuild in progress.
        self._build_reporter: TuiReporter | None = None
        self._build_started = 0.0
        self._build_tick = 0
        self._building = False

    def compose(self) -> ComposeResult:
        yield Static("STORED USERNAMES", classes="pane-title")
        with Grid(id="results-body"):
            yield UsernameList(id="username-list", cursor_type="row")
            with Vertical(id="result-detail"):
                # Always visible, whichever section is showing: who this is and
                # when it was scanned is context for all three.
                yield Static(id="detail-header")
                yield Tabs(
                    Tab("ACCOUNTS", id=TAB_ACCOUNTS),
                    Tab("UNRESOLVED", id=TAB_UNRESOLVED),
                    Tab("PROFILE", id=TAB_PROFILE),
                    id="detail-tabs",
                )
                # Above the tables, not below them: a key is read before the
                # rows it explains or it is read too late. It belongs to the
                # section rather than to the pane, so it sits under the strip
                # that switches them and changes with it.
                yield Static(id="detail-legend")
                # Each section owns its own scrolling, and only one is mounted
                # visible at a time -- which is the whole fix for the double
                # scrollbar.
                with ContentSwitcher(initial=SEC_ACCOUNTS, id="detail-switch"):
                    yield DataTable(id=SEC_ACCOUNTS, cursor_type="row")
                    yield DataTable(id=SEC_UNRESOLVED, cursor_type="row")
                    with VerticalScroll(id=SEC_PROFILE):
                        yield Static(id="detail-profile")
                        # Shown only when there is no profile to display. The
                        # section otherwise stays a viewer.
                        with Vertical(id="profile-actions"):
                            yield Static(id="profile-anchor-line")
                            # Progress reports HERE, not in the scan tab's
                            # ACTIVITY log: that is a different tab, and being
                            # told to go and watch somewhere else is not
                            # feedback. An anchored rebuild loads the model,
                            # which has been measured at 187s cold, so silence
                            # is the one thing this must not do.
                            yield Static(id="profile-status")
                            with Grid(id="profile-buttons"):
                                yield Static()
                                yield Button("Anchors", id="profile-anchors")
                                yield Button(
                                    "Build profile",
                                    variant="primary",
                                    id="profile-build",
                                )

    def on_mount(self) -> None:
        table = self.query_one("#username-list", UsernameList)
        # Widths chosen to fit the fixed column the stylesheet gives this list,
        # padding and scrollbar included. At their previous size the last column
        # was clipped to "si" and its number could not be read.
        #
        # 13 rather than 15 because the delete control took two cells from
        # somewhere and this was the cheapest place: the alternative was two
        # more cells of pane, which costs the section strip opposite a tab --
        # see the arithmetic in theme.py. A name too long for 13 ellipsizes;
        # a tab that has scrolled out of the strip is simply gone.
        table.add_column("username", key="username", width=13)
        # "found" before "sites": the hit count is what someone is scanning the
        # list for, and the total is context for it. Reversed, the eye lands on
        # the larger, less interesting number first on every row.
        table.add_column("found", key="found", width=5)
        table.add_column("sites", key="sites", width=5)
        # The delete control's column: unlabelled, as wide as the control, and
        # empty on every row the pointer is not on. Added LAST, so it sits at
        # the end of the row: a control that acts on the whole row belongs after
        # the facts about that row, not in front of them.
        table.add_column("", key=UsernameList.DELETE_COLUMN, width=3)

        accounts = self.query_one(f"#{SEC_ACCOUNTS}", DataTable)
        accounts.add_column("", key="mark", width=2)
        accounts.add_column("site", key="site", width=18)
        accounts.add_column("url", key="url")

        unresolved = self.query_one(f"#{SEC_UNRESOLVED}", DataTable)
        unresolved.add_column("", key="mark", width=2)
        unresolved.add_column("site", key="site", width=18)
        unresolved.add_column("why", key="why")

        self._set_building(False)
        # One timer for the pane; it costs an attribute check per tick when
        # nothing is building.
        self.set_interval(REDRAW_INTERVAL, self._tick_build_status)
        self._load()

    # -- loading ------------------------------------------------------------

    def action_reload(self) -> None:
        self._load()

    def select_username(self, username: str) -> None:
        """Open a particular username, reloading first if it is not listed yet.

        Used when the scan pane sends someone here instead of re-scanning. The
        row may not be drawn yet, so the name is remembered and applied by the
        loader rather than looked up now and missed.
        """
        self._pending_selection = username
        self._load()

    @work(exclusive=True, group="results-list")
    async def _load(self) -> None:
        db = await SherlockDB.create(str(default_database_path()))
        try:
            self._listings = await db.list_usernames()
        finally:
            await db.close()
        self._fill_list()

    def _fill_list(self) -> None:
        table = self.query_one("#username-list", UsernameList)
        table.clear()
        if not self._listings:
            self._set_detail(
                Text(
                    "Nothing scanned yet.\n\n"
                    "Run a scan on the SCAN tab and results will appear here.",
                    style="dim italic",
                )
            )
            return
        for listing in self._listings:
            table.add_row(
                # Text(), because a username is user data and Rich reads markup
                # in cells -- the same reason `show` wraps its own.
                Text(listing.username, overflow="ellipsis", no_wrap=True),
                Text(
                    str(listing.claimed_sites),
                    style="bold green" if listing.claimed_sites else "dim",
                ),
                Text(str(listing.total_sites), style="dim"),
                # The delete control's cell, empty until the pointer is on this
                # row. Its own `Text`, not the shared blank: a cell's value is
                # kept by the table, and one object shared by every row is one
                # edit away from every row changing together.
                NO_DELETE.copy(),
                key=listing.username,
            )
        # Open on whichever row was asked for, otherwise the first. The list is
        # sorted most-recent-first, so the top row is almost always the one
        # someone came back to look at.
        wanted = self._pending_selection
        self._pending_selection = None
        names = [listing.username for listing in self._listings]
        row = names.index(wanted) if wanted in names else 0
        table.move_cursor(row=row)
        self._select(names[row])

    @on(DataTable.RowHighlighted, "#username-list")
    def _highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.row_key is not None and event.row_key.value:
            self._select(str(event.row_key.value))

    def action_toggle_sources(self) -> None:
        """Swap the compact source summary for full URLs, and back.

        The same choice `show --sources` offers, for the same reason: "3 sites:
        A, B +1" is what you want while reading, and the full URLs are what you
        want while checking.
        """
        self._show_sources = not self._show_sources
        self._redraw_profile()

    def action_toggle_notes(self) -> None:
        """Show the diagnostic notes in place, or fold them back to a count.

        The command line answers this with `--verbose` and a second run. In an
        app the equivalent is a key, and the profile block names it -- a hint
        that points at a flag nobody here can type is worse than no hint.
        """
        self._show_notes = not self._show_notes
        self._redraw_profile()

    def _redraw_profile(self) -> None:
        """Re-render the profile from the record already in hand.

        Deliberately not `_select`, which re-reads the database: nothing on
        disk changed, only how this pane is drawing it, and a toggle that costs
        two queries is a toggle that feels slow.
        """
        if self._record is not None:
            self.query_one("#detail-profile", Static).update(
                self._profile_block(self._record)
            )

    def action_export(self) -> None:
        """Write the selected username's record to JSON in the working directory.

        A screen cannot be piped, which is the one real thing the CLI has that a
        UI does not -- so without an export the UI is a place where evidence can
        be looked at and never taken away, and every investigation ends by
        retyping the command line anyway.

        Deliberately the same payload `sherlock show --json` produces, from the
        same function, so a file written here and one written there are the same
        file. `ensure_ascii` is on for the reason that command documents:
        profiles carry names outside cp1252 and Windows encodes redirected
        output with it.
        """
        if self._record is None or not self._record.get("known"):
            self.notify("Nothing to export.", severity="warning")
            return
        from sherlock_project.show import _as_json

        target = Path.cwd() / f"{self._record['username']}.json"
        try:
            target.write_text(_as_json([self._record]), encoding="utf-8")
        except OSError as error:
            self.notify(f"Could not write {target}: {error}", severity="error")
            return
        self.notify(f"Wrote {target}")

    def action_delete_username(self) -> None:
        """Erase the selected username: the keyboard half of the row's ✕.

        One action with two ways in, not two actions. The key asks about the
        SELECTED row, the control asks about the row under the pointer, and
        both land in `_confirm_delete` with the same question.

        Reads the selection rather than the loaded record, so it works in the
        moment between picking a row and its detail arriving -- the detail is a
        separate database read, and a key that does nothing for a beat looks
        broken.
        """
        if self._selected is None:
            self.notify("Nothing selected to delete.", severity="warning")
            return
        self._confirm_delete(self._selected)

    @on(UsernameList.DeleteRequested, "#username-list")
    def _delete_from_row(self, event: UsernameList.DeleteRequested) -> None:
        """The ✕ on a row, which is not necessarily the selected row.

        That is the point of having it: removing a username you can see should
        not require opening it first, and opening it is what the keyboard path
        makes you do.
        """
        self._confirm_delete(event.username)

    def _confirm_delete(self, username: str) -> None:
        """Ask, with figures, before erasing everything stored for a username.

        A scan is a dossier on a person, and there was no way to remove one --
        scanning the wrong name left a permanent local record with no in-app
        way to undo it. For a tool whose subject is people that is a privacy
        function, not a convenience.

        The confirmation names the username and counts what will go, because
        "are you sure?" with no figures is a question nobody can answer. It is
        the only destructive action in the app and it is the reason it is the
        only one that asks.

        The figures come from the LISTING rather than from the loaded detail.
        The ✕ acts on the row under the pointer, whose detail is not loaded and
        may never be; a dialog that counted the selected username's results
        while naming another one would be worse than not asking at all.
        """
        listing = next(
            (item for item in self._listings if item.username == username), None
        )
        if listing is None:
            self.notify("Nothing selected to delete.", severity="warning")
            return

        profile = "and its AI profile " if listing.has_profile else ""
        detail = (
            f"Delete everything stored for {username!r}?\n\n"
            f"{count_of(listing.total_sites, 'site result')} {profile}will be "
            f"removed, including {count_of(listing.claimed_sites, 'account')} "
            f"found.\n"
            f"This cannot be undone, and the scan itself cannot be recovered "
            f"without running it again."
        )

        def erase(confirmed: bool | None) -> None:
            if confirmed:
                self._erase(username)

        self.app.push_screen(
            ConfirmScreen(f"Delete {username}", detail), erase
        )

    @work(exclusive=True, group="results-delete")
    async def _erase(self, username: str) -> None:
        db = await SherlockDB.create(str(default_database_path()))
        try:
            removed = await db.delete_username(username)
        finally:
            await db.close()

        if self._selected == username:
            # Forget the record too, or the detail pane keeps drawing a username
            # that no longer exists until something else happens to reload it.
            self._record = None
            self._selected = None
        else:
            # Deleting the row the POINTER was on must not move the reader off
            # the row they were reading: the reload reopens the first username
            # unless it is told which one to go back to.
            self._pending_selection = self._selected
        self.notify(f"Deleted {username} ({count_of(removed, 'result')}).")
        self.action_reload()

    def _select(self, username: str) -> None:
        self._selected = username
        self._load_detail(username)

    @work(exclusive=True, group="results-detail")
    async def _load_detail(self, username: str) -> None:
        # `_collect` from show.py rather than a second set of queries: it
        # already decides what "unresolved" means and how a stored profile that
        # no longer validates is reported. Two answers to those questions is one
        # too many.
        from sherlock_project.show import _collect

        db = await SherlockDB.create(str(default_database_path()))
        try:
            record = await _collect(db, username)
        finally:
            await db.close()
        self._record = record
        self._show_record(record)

    def _set_detail(self, renderable: Any) -> None:
        """Show a bare message in place of a record."""
        self.query_one("#detail-header", Static).update(renderable)
        self.query_one(f"#{SEC_ACCOUNTS}", DataTable).clear()
        self.query_one(f"#{SEC_UNRESOLVED}", DataTable).clear()
        self.query_one("#detail-profile", Static).update("")
        self._account_urls.clear()
        self._section_statuses.clear()
        self._redraw_legend()
        self._set_tab_counts(accounts=0, unresolved=0)

    def _set_tab_counts(self, *, accounts: int, unresolved: int) -> None:
        """Put the counts on the tab labels.

        This is what makes switching safe. The stacked layout kept the
        unresolved count next to the account list because it changes what that
        list means -- six sites that never answered is not the same as six
        sites where nobody was home. Behind a bare tab that number would be
        invisible until someone thought to look, so it goes on the tab.
        """
        labels = {
            TAB_ACCOUNTS: f"ACCOUNTS {accounts}" if accounts else "ACCOUNTS",
            TAB_UNRESOLVED: (
                f"UNRESOLVED {unresolved}" if unresolved else "UNRESOLVED"
            ),
        }
        tabs = self.query_one("#detail-tabs", Tabs)
        for tab_id, label in labels.items():
            tabs.query_one(f"#{tab_id}", Tab).label = label

    @on(Tabs.TabActivated, "#detail-tabs")
    def _switch_section(self, event: Tabs.TabActivated) -> None:
        section = SECTION_FOR_TAB.get(event.tab.id or "")
        if section is not None:
            self.query_one("#detail-switch", ContentSwitcher).current = section
            self._redraw_legend()

    def _redraw_legend(self) -> None:
        """Name the symbols the section on screen is actually using.

        Per section, because the sections do not share a vocabulary, and a key
        only earns its row where symbols contend. That is UNRESOLVED, and only
        UNRESOLVED: it is the one place the distinction between "the site
        blocked us", "the rules did not decide" and "the site would never have
        allowed this name" is drawn -- the tool's central claim, made in two
        cells of a table nobody has been given a glossary for. ACCOUNTS is one
        symbol repeated, and PROFILE has none at all.

        Hidden rather than blanked when there is nothing to explain. A blank
        line still occupies a row, and this pane is short enough that every row
        spent on chrome is a finding not shown.
        """
        current = self.query_one("#detail-switch", ContentSwitcher).current
        statuses = self._section_statuses.get(current or "", [])
        legend = self.query_one("#detail-legend", Static)
        legend.display = bool(statuses)
        if statuses:
            # "key" spelled out, dim, in front. Without it the line is a
            # symbol followed by a word directly above a table of symbols
            # followed by words -- which is to say, it reads as the first row.
            legend.update(
                Text.assemble(("key  ", "dim"), status_key(statuses))
            )

    def action_next_section(self) -> None:
        self.query_one("#detail-tabs", Tabs).action_next_tab()

    def action_prev_section(self) -> None:
        self.query_one("#detail-tabs", Tabs).action_previous_tab()

    # -- rendering ----------------------------------------------------------

    def _show_record(self, record: dict[str, Any]) -> None:
        if not record.get("known"):
            self._set_detail(
                Text("Nothing stored for this username.", style="dim italic")
            )
            return

        unresolved = record.get("unresolved") or []
        accounts = record["accounts"]

        self.query_one("#detail-header", Static).update(
            Text.assemble(
                (record["username"], "bold cyan"),
                ("\n", ""),
                (
                    (
                        f"last scanned {record['last_scanned_at']}  ·  "
                        f"{count_of(record['sites_checked'], 'site')} checked"
                    ),
                    "dim",
                ),
            )
        )
        self._fill_accounts(accounts)
        self._fill_unresolved(unresolved)
        self._redraw_legend()
        self._set_tab_counts(accounts=len(accounts), unresolved=len(unresolved))
        self._seed_build_anchors(record)
        self.query_one("#detail-profile", Static).update(
            self._profile_block(record)
        )
        self._redraw_profile_actions(record)

    def _fill_unresolved(self, unresolved: list[dict[str, Any]]) -> None:
        """The sites that gave no answer, listed rather than only counted.

        There was no in-app equivalent of `show --unresolved` before this, and a
        UI needs one more than the CLI does -- there is no pipe to fall back on.
        Absence here is not evidence of absence, which is the whole reason these
        are kept apart from the accounts instead of quietly dropped.
        """
        table = self.query_one(f"#{SEC_UNRESOLVED}", DataTable)
        table.clear()
        self._section_statuses[SEC_UNRESOLVED] = []
        if not unresolved:
            table.add_row(
                Text(""),
                Text("—", style="dim"),
                Text("every site gave an answer", style="dim italic"),
            )
            return
        drawn: list[QueryStatus] = []
        for entry in unresolved:
            # The stored status, not a prefix match on the sentence written
            # from it. Reading the symbol back out of its own explanation meant
            # anything that did not start with "blocked" was drawn as
            # inconclusive -- so a username the site's own rules reject, which
            # `show` reports as "username format rejected" and has its own ✕,
            # arrived here wearing the symbol for "we could not tell". That is
            # exactly the conflation the unresolved list exists to prevent, and
            # a key naming the symbols would have printed the wrong word beside
            # it with the same confidence.
            status = status_from_name(entry.get("status"))
            style = status_style(status)
            drawn.append(status)
            detail = entry["reason"]
            if entry.get("transport") == "http":
                # Often the whole explanation for an inconclusive result, so it
                # goes before the symptom rather than behind it.
                detail = f"{detail}; no browser"
            if entry.get("context"):
                detail = f"{detail}; {entry['context']}"
            table.add_row(
                Text(style.glyph, style=style.style),
                Text(str(entry["site_name"]), overflow="ellipsis", no_wrap=True),
                Text(detail, style="dim", overflow="ellipsis", no_wrap=True),
            )
        self._section_statuses[SEC_UNRESOLVED] = drawn

    def _fill_accounts(self, accounts: list[dict[str, Any]]) -> None:
        table = self.query_one(f"#{SEC_ACCOUNTS}", DataTable)
        table.clear()
        self._account_urls.clear()
        # No entry in `_section_statuses`, which is what keeps the key off this
        # section. Every row here is a hit, so the mark column is one symbol
        # repeated down a list the tab already calls ACCOUNTS -- there is
        # nothing for a key to disambiguate, and it would be spending a row of
        # findings to say so. A key earns its row where symbols contend.
        if not accounts:
            # One row rather than an empty table, so the section reads as
            # answered rather than as still loading.
            table.add_row(Text(""), Text("—", style="dim"),
                          Text("no accounts found", style="dim italic"))
            return

        found = status_style(QueryStatus.CLAIMED)
        for index, account in enumerate(accounts):
            note = []
            if account.get("confidence") and account["confidence"] != "Confirmed":
                note.append(str(account["confidence"]))
            # A hit found without a browser is weaker evidence than one found
            # with it, and this is the only place months later that says so --
            # the same annotation `show` makes, for the same reason.
            if account.get("transport") == "http":
                note.append("no browser")
            key = str(index)
            self._account_urls[key] = str(account["url"])
            table.add_row(
                Text(found.glyph, style=found.style),
                Text(str(account["site_name"]), overflow="ellipsis", no_wrap=True),
                Text.assemble(
                    (str(account["url"]), ""),
                    (f"  [{'; '.join(note)}]" if note else "", "dim"),
                ),
                key=key,
            )

    @on(DataTable.RowSelected, f"#{SEC_ACCOUNTS}")
    def _open_account(self, event: DataTable.RowSelected) -> None:
        """Enter on a stored account opens it.

        This is where the action matters most: the live feed scrolls past, but
        this list is what someone comes back to days later, and retyping a URL
        off a terminal is exactly the friction the pane exists to remove.
        """
        url = self._account_urls.get(event.row_key.value or "")
        if not url:
            self.notify("No link for that row.", severity="warning")
            return
        webbrowser.open(url, new=2)

    def _set_building(self, building: bool) -> None:
        """Swap the controls for a status line while the work runs.

        The buttons go away rather than being left pressable: a second Rebuild
        on top of the first would start a competing synthesis over the same
        rows, and "nothing appeared to happen" is exactly why someone presses
        twice.
        """
        self._building = building
        self.query_one("#profile-buttons").display = not building
        status = self.query_one("#profile-status", Static)
        status.display = building
        if building:
            status.update(Text("Starting…", style="dim"))

    def _tick_build_status(self) -> None:
        """Draw what the rebuild is doing, ten times a second.

        Reuses the scan screen's phase line, so a model loading here looks
        exactly like a model loading there -- one vocabulary for one event,
        whichever screen is waiting on it.
        """
        reporter = self._build_reporter
        if reporter is None:
            return
        self._build_tick += 1

        text = Text()
        for phase in reporter.phases:
            text.append_text(
                phase_line(
                    phase.label, phase.state, elapsed=phase.elapsed,
                    tick=self._build_tick,
                )
            )
            text.append("\n")
        if not text.plain:
            # No model needed -- an unanchored rebuild is a merge and never
            # reaches a phase. Say something anyway; a blank line while working
            # is the silence this replaced.
            elapsed = perf_counter() - self._build_started
            text.append(
                f"{spinner(self._build_tick)} merging evidence  "
                f"{elapsed_label(elapsed)}",
                style="cyan",
            )
        notes = reporter.snapshot_log()
        if notes:
            text.append(notes[-1].plain.strip(), style="dim")
        self.query_one("#profile-status", Static).update(text)

    def _seed_build_anchors(self, record: dict[str, Any]) -> None:
        """Carry a profile's own anchors into a rebuild of it.

        This is the fix for a defect the CLI still has: `--ai-synthesize-only`
        takes anchors from the command line and never from the profile it is
        overwriting, so rebuilding without re-typing `--anchor` discards them.
        `has_anchors` goes false, synthesis falls through to the aggregate path,
        and the identity resolution is not recomputed -- it is abandoned. On
        real data that turned 4 fields / 7 values / 2 anchors into 11 fields /
        73 values / 0 anchors: bigger, and worse.

        Seeding them here makes the default the safe one, and the line above the
        button shows what will be used -- so it is visible state rather than a
        hidden default, which is the part a flag could not have given us.

        Re-seeded only when the selected username changes, or edits made for one
        username would be silently thrown away by clicking back to it.
        """
        username = str(record.get("username") or "")
        if username == self._anchors_for:
            return
        self._anchors_for = username
        profile = record.get("profile")
        stored = list(getattr(profile, "anchors", []) or [])
        self._build_anchors = stored

    @staticmethod
    def _extraction_count(record: dict[str, Any]) -> int:
        """How many sites have stored Pass 1 evidence.

        This is the precondition for building anything. Pass 2 merges stored
        extractions -- it does not read pages -- so a username scanned WITHOUT
        analysis has nothing to synthesise, and offering a Build button there
        would produce an empty profile and look broken rather than say why.
        """
        return sum(
            int(entry.get("count") or 0)
            for entry in (record.get("extraction_models") or [])
        )

    def _redraw_profile_actions(self, record: dict[str, Any]) -> None:
        """Offer the action that fits what this username actually has.

        Three states, three answers, because a single always-on "Build" button
        would be wrong in two of them:
          - a profile already exists  -> nothing to offer; this is a viewer
          - no stored extractions     -> building is impossible; the fix is to
                                         scan again with analysis on
          - extractions stored        -> building is genuinely available
        """
        actions = self.query_one("#profile-actions", Vertical)
        hint = self.query_one("#profile-anchor-line", Static)
        build = self.query_one("#profile-build", Button)
        anchors = self.query_one("#profile-anchors", Button)

        profile = record.get("profile")
        has_profile = profile is not None
        actions.display = bool(record.get("known"))
        if not actions.display:
            return

        # Rebuilding pass 2 belongs HERE, beside the profile it replaces --
        # not on the scan tab, which scans nothing to do it, and not on a tab of
        # its own. The moment you want different anchors is the moment you are
        # looking at a profile that mixed two people together.
        evidence = self._extraction_count(record)
        if not evidence:
            # Nothing to merge. Say what is missing and what fixes it, rather
            # than offering a button that would build an empty profile.
            hint.update(
                Text(
                    "No AI evidence stored — this username was scanned without "
                    "analysis.\nScan it again with analysis on to collect what "
                    "a profile is built from.",
                    style="dim",
                )
            )
            build.label = "Scan with analysis"
            anchors.display = False
            return

        anchors.display = True
        build.label = "Rebuild profile" if has_profile else "Build profile"

        line = Text()
        if has_profile:
            line.append(
                f"Re-runs the second AI pass over the same evidence from "
                f"{count_of(evidence, 'site')}. Nothing is re-fetched.\n"
            )
        else:
            line.append(f"Evidence from {count_of(evidence, 'site')} is ready.\n")

        if self._build_anchors:
            fields = ", ".join(a.field for a in self._build_anchors[:3])
            extra = len(self._build_anchors) - 3
            line.append(
                f"anchored to {fields}{f' +{extra}' if extra > 0 else ''} — "
                f"needs the local model",
                style="dim",
            )
        elif has_profile and getattr(profile, "mode", "") == "anchored":
            # The downgrade this pane exists to prevent. Rebuilding an anchored
            # profile with no anchors does not recompute the identity -- it
            # ABANDONS it, and the result looks bigger while being worse: a
            # resolved identity replaced by a merge of every name every site
            # showed. Measured at 4 fields/7 values becoming 11 fields/73.
            line.append(
                "This profile is anchored, but no anchors are set — rebuilding "
                "now would replace it with an unresolved merge.",
                style="yellow",
            )
        else:
            # The unanchored path needs no model at all: aggregate synthesis
            # merges stored extractions and never calls one. Worth saying,
            # because "build a profile" otherwise reads as a slow operation.
            line.append(
                "No anchors — builds instantly, and the result describes "
                "anyone sharing this username.",
                style="dim",
            )
        hint.update(line)

    @on(Button.Pressed, "#profile-anchors")
    def _edit_build_anchors(self) -> None:
        """The SAME editor the scan pane uses, reached from a second place.

        Not a second anchor CRUD surface: one editor with two entry points is
        fine, two implementations of the same list is the two-homes problem
        that kept the AI tab, the trust picker and the re-scan toggle from
        existing.
        """
        def adopt(updated: list[IdentityAnchor] | None) -> None:
            if updated is None:
                return
            self._build_anchors = updated
            if self._record is not None:
                self._redraw_profile_actions(self._record)

        self.app.push_screen(AnchorScreen(self._build_anchors), adopt)

    @on(Button.Pressed, "#profile-build")
    def _build_profile(self) -> None:
        record = self._record
        if record is None or not record.get("known"):
            return
        username = str(record["username"])

        if not self._extraction_count(record):
            # The button is the pointer, not the action: this pane cannot scan.
            self.post_message(self.ScanWithAnalysis(username))
            return
        self._synthesize(username, rebuild=record.get("profile") is not None)

    class ScanWithAnalysis(Message):
        """Asked to go and collect the evidence a profile needs.

        A message rather than a tab switch: which tab scans is the app's
        business, the same way `ScanPane.ShowStored` works in the other
        direction.
        """

        def __init__(self, username: str) -> None:
            super().__init__()
            self.username = username

    @work(exclusive=True, group="results-build")
    async def _synthesize(self, username: str, *, rebuild: bool = False) -> None:
        """Build or rebuild the profile from evidence already on disk.

        `run_synthesis_only` is the CLI's own `--ai-synthesize-only` path, used
        rather than reimplemented -- it loads a model only when anchors make one
        necessary, and closes both the model and the database whatever happens.

        `force` follows the button, and the two cases really are different.
        A first BUILD passes False: there is nothing to overwrite, and if a
        cached summary somehow matches then reusing it is correct. A REBUILD
        passes True, because someone pressed a button labelled rebuild -- unchanged
        anchors would otherwise hit the cache and the button would appear to do
        nothing at all, which is a worse failure than the work being redone.
        """
        from sherlock_project.sherlock import run_synthesis_only

        # A reporter, so the work narrates itself. It was called with none at
        # all before, which meant model loading and synthesis progress went
        # nowhere and the only feedback was a toast that had already vanished.
        reporter = TuiReporter()
        self._build_reporter = reporter
        self._build_started = perf_counter()
        self._set_building(True)
        try:
            await run_synthesis_only(
                usernames=[username],
                force=rebuild,
                inline_anchors=list(self._build_anchors),
                reporter=reporter,
            )
        except Exception as error:
            # Broad on purpose, as everywhere else here: a failed synthesis
            # must not take the pane down with it. Written into the status line
            # rather than a toast -- a failure that disappears after five
            # seconds is a failure nobody can act on.
            self._set_building(False)
            self.query_one("#profile-status", Static).update(
                Text(f"Could not build profile: {error}", style="red")
            )
            return
        finally:
            self._build_reporter = None

        self._set_building(False)
        self.notify(f"Profile built for {username}.")
        self._select(username)

    def _profile_block(self, record: dict[str, Any]) -> Any:
        title = Text("AI PROFILE", style="bold")
        if record.get("profile_unreadable"):
            return Group(
                title,
                Text(
                    "A profile is stored but no longer matches the current "
                    "format. Rebuild it from the CLI:\n"
                    f"  sherlock {record['username']} --ai-synthesize-only",
                    style="yellow",
                ),
            )
        profile = record.get("profile")
        if profile is None:
            return Group(
                title,
                Text(
                    "No profile stored. Scanning with a model configured "
                    "builds one.",
                    style="dim italic",
                ),
            )
        return Group(
            title,
            Text(f"built {record['profile_updated_at']}", style="dim"),
            Text(),
            render_profile_text(
                profile,
                show_sources=self._show_sources,
                show_notes=self._show_notes,
            ),
        )


def render_profile_text(
    profile: Any,
    *,
    show_sources: bool = False,
    show_notes: bool = False,
) -> Text:
    """The stored profile, drawn by the renderer both other surfaces use.

    `render_profile` prints to a console rather than returning a renderable, so
    this captures it. That is deliberately not a third implementation: the
    layout, the CONFIDENT/UNSURE split and the source summarising are decisions
    that must stay identical across `show --profile`, the end of an `--ai` run
    and this pane. A pane that drew its own would be free to reorder or cap
    values, which is exactly what the renderer's docstring forbids.

    ANSI is preserved and reparsed rather than stripped, so the confidence
    colouring survives the round trip.
    """
    from sherlock_project.notify import TerminalReporter

    buffer = StringIO()
    console = Console(
        file=buffer,
        width=PROFILE_RENDER_WIDTH,
        color_system="truecolor",
        highlight=False,
        force_terminal=True,
    )
    reporter = TerminalReporter(
        console=console, error_console=console, verbose=show_notes
    )
    reporter.render_profile(
        profile,
        show_sources=show_sources,
        # This surface has a key, not a flag. Telling someone to "run with
        # --verbose" inside a running app points at a command line they are not
        # at, which is how CLI advice ended up on screen with no way to act on
        # it.
        notes_hint="press v to read them.",
    )
    return Text.from_ansi(buffer.getvalue())
