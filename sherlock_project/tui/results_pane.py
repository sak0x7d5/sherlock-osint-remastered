"""The results pane: pick a username on the left, read what is stored on the right.

Master/detail, because the question is always "this one, tell me more" and a
single flat list cannot answer it without either truncating the detail or
hiding the names.

**The detail is two switched sections, not one long scroll.** This started as
a single stacked column on the argument that the sites and the profile are read
together. Stacking them was wrong for a reason that only shows
up with real data: a `DataTable` is itself a scrollable viewport, so a table
inside a scrolling column produces TWO vertical scrollbars side by side, plus a
horizontal one when the URLs are long. Forty accounts made the pane look broken,
and it also meant scrolling past all forty to reach the profile.

Exactly one scroll region is visible at a time now. The switcher is a `Tabs`
strip driving a `ContentSwitcher` rather than a nested `TabbedContent`, because
a second full tab bar directly under the first reads as two competing
navigations; a compact strip reads as what it is, a section switch inside one
place.

**One table for what was found and what could not be decided.** These were two
sections, ACCOUNTS and UNRESOLVED, and `show` still keeps the two lists apart
for a reason worth restating: they answer different questions, and merging them
is what let silence read as absence in the first place. What made that merge
dangerous was dropping the distinction, not showing the rows together, and
nothing is dropped here -- every row wears the status it was stored with, the
key names all four, and the line above the table reports the unresolved count
even while `found only` is filtering those rows out. Two sections cost more
than they protected: you could sit on ACCOUNTS and never learn UNRESOLVED
existed, and one table with 183 unanswered rows in it cannot be read that way.

**The counts moved off the tab labels onto that line.** They were on the tabs
because switching would otherwise hide one section's number from the other; one
table makes that arrangement unnecessary, and the numbers now sit next to the
control that acts on them.

**A symbol column comes with a key.** The mark column is two cells wide and its
header is blank, which left the distinction this tool exists to make -- blocked
is not inconclusive is not rejected -- drawn in symbols nobody had been shown a
glossary for. The live feed on the scan pane does not have that problem because
it prints the word beside every symbol; a table has no room for that, so the
glossary goes in the header band, in the half of it that was empty at every
width. It is the one bordered block on this screen, deliberately: a key is not
data, and in an otherwise flat pane the border is what says so at a glance.

Not beside the tabs, which is where the spare width looks like it is. Measured,
that space is not spare: `Tabs` is a scrolling strip, so squeezed it drops tabs
rather than wrapping or ellipsizing them, silently and from the left. A key
sharing that row needed a 126-column terminal to leave three tabs intact.

The border costs the two rows it occupies, and the box costs 30 cells of width
that the header beside it does not have on a narrow terminal -- so below
`KEY_MIN_DETAIL_WIDTH` the key is taken off screen rather than wrapping the
timestamp next to it into a column of fragments.

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
from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Grid, Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import (
    Button,
    ContentSwitcher,
    DataTable,
    Static,
    Tab,
    Tabs,
)

# Section ids. The tab and the pane it selects need SEPARATE ids even though
# they are the same section: two widgets sharing one id makes `query_one`
# ambiguous, and which of the two it returns depends on walk order rather than
# on what the caller asked for. `ContentSwitcher` selects a child by id and
# `Tabs` reports the activated tab by id, so the mapping below is what connects
# them -- one table, rather than a string transformation at each call site.
SEC_SITES = "sec-sites"
SEC_PROFILE = "sec-profile"

TAB_SITES = "tab-sites"
TAB_PROFILE = "tab-profile"

SECTION_FOR_TAB = {
    TAB_SITES: SEC_SITES,
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

# What the key names. Every status a stored record can put in the SITES table,
# which is all of them except AVAILABLE: `show` does not list absent sites at
# all, so a key offering `· absent` would name a symbol that cannot appear.
#
# Fixed rather than derived from the rows on screen. The key is a glossary for
# the app's vocabulary, not a summary of one record -- and a bordered block
# that changes height between usernames would move the table under it.
# Detail-pane width below which the key is taken off screen. The box is 30
# cells and the line it sits beside is 32 (`last scanned` plus a timestamp), so
# under their sum the header does not shorten -- it WRAPS. Measured at an
# 80-column terminal that is six lines of broken-up timestamp where there were
# two, and the table is pushed down by all of them. A glossary is worth a lot
# less than being able to read the record it is a glossary for.
KEY_MIN_DETAIL_WIDTH = 64

KEY_STATUSES: tuple[QueryStatus, ...] = (
    QueryStatus.CLAIMED,
    QueryStatus.UNKNOWN,
    QueryStatus.WAF,
    QueryStatus.ILLEGAL,
)


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
        Binding("f", "toggle_found_only", "found only"),
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
        # Row key -> the URL that row is about, so Enter can open it. Every
        # row, not just the hits: an unresolved site has a URL too, and "let me
        # go and look at the one that blocked us" is the obvious next move from
        # a row that says nothing could be decided.
        self._row_urls: dict[str, str] = {}
        # Whether the SITES table is showing hits only. On by default, so the
        # pane still opens on the answer to "what did I find" -- the unresolved
        # rows are one keypress away and the line above the table says how many
        # are behind it, which is the part that keeps silence from reading as
        # absence.
        self._found_only = True
        # Live state for a rebuild in progress.
        self._build_reporter: TuiReporter | None = None
        self._build_started = 0.0
        self._build_tick = 0
        self._building = False

    def compose(self) -> ComposeResult:
        yield Static("STORED USERNAMES", classes="pane-title")
        with Grid(id="results-body"):
            yield DataTable(id="username-list", cursor_type="row")
            with Vertical(id="result-detail"):
                # Identity left, key right. The right half of this band was
                # empty at every width -- a username and a timestamp do not
                # fill 90 cells -- so the glossary goes in the space the header
                # was already paying for rather than in a row of its own.
                with Grid(id="detail-head"):
                    # Always visible, whichever section is showing: who this is
                    # and when it was scanned is context for both.
                    yield Static(id="detail-header")
                    yield Static(id="detail-keys")
                yield Tabs(
                    Tab("SITES", id=TAB_SITES),
                    Tab("PROFILE", id=TAB_PROFILE),
                    id="detail-tabs",
                )
                # The filter, and what it is hiding. Only on SITES: PROFILE has
                # nothing to filter, and a control that cannot do anything is
                # worse than an absent one.
                with Grid(id="sites-controls"):
                    yield Button(id="toggle-found-only", classes="toggle")
                    yield Static(id="sites-counts")
                # Each section owns its own scrolling, and only one is mounted
                # visible at a time -- which is the whole fix for the double
                # scrollbar.
                with ContentSwitcher(initial=SEC_SITES, id="detail-switch"):
                    yield DataTable(id=SEC_SITES, cursor_type="row")
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
        table = self.query_one("#username-list", DataTable)
        # Widths chosen to fit the fixed 34-cell column the stylesheet gives
        # this list, padding included. At their previous size the last column
        # was clipped to "si" and its number could not be read.
        table.add_column("username", key="username", width=15)
        # "found" before "sites": the hit count is what someone is scanning the
        # list for, and the total is context for it. Reversed, the eye lands on
        # the larger, less interesting number first on every row.
        table.add_column("found", key="found", width=5)
        table.add_column("sites", key="sites", width=5)

        sites = self.query_one(f"#{SEC_SITES}", DataTable)
        sites.add_column("", key="mark", width=2)
        sites.add_column("site", key="site", width=18)
        # One column for two kinds of answer: the URL where a hit was found,
        # the reason nothing was decided otherwise. Neither "url" nor "why" is
        # true of the other half, so the header names what the column IS rather
        # than what half of it happens to hold.
        sites.add_column("detail", key="detail")

        # Drawn once. The key is the app's vocabulary, not this record's, so
        # nothing that happens to the data can change it.
        keys = self.query_one("#detail-keys", Static)
        keys.border_title = "keys"
        keys.update(status_key(KEY_STATUSES, columns=2))
        self._redraw_found_only()
        self._fit_keys()

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
        table = self.query_one("#username-list", DataTable)
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
        """Erase everything stored for the selected username, after confirming.

        A scan is a dossier on a person, and there was no way to remove one --
        scanning the wrong name left a permanent local record with no in-app
        way to undo it. For a tool whose subject is people that is a privacy
        function, not a convenience.

        The confirmation names the username and counts what will go, because
        "are you sure?" with no figures is a question nobody can answer. It is
        the only destructive action in the app and it is the reason it is the
        only one that asks.
        """
        record = self._record
        if record is None or not record.get("known"):
            self.notify("Nothing selected to delete.", severity="warning")
            return

        username = str(record["username"])
        found = record.get("accounts_found") or 0
        checked = record.get("sites_checked") or 0
        profile = "and its AI profile " if record.get("profile") is not None else ""
        detail = (
            f"Delete everything stored for {username!r}?\n\n"
            f"{count_of(checked, 'site result')} {profile}will be removed, "
            f"including {count_of(found, 'account')} found.\n"
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

        # Forget the record too, or the detail pane keeps drawing a username
        # that no longer exists until something else happens to reload it.
        self._record = None
        self._selected = None
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
        self.query_one(f"#{SEC_SITES}", DataTable).clear()
        self.query_one("#detail-profile", Static).update("")
        self._row_urls.clear()
        self._redraw_counts(found=0, unresolved=0)

    def _redraw_counts(self, *, found: int, unresolved: int) -> None:
        """Say what the table holds, and what the filter is holding back.

        These numbers used to live on the tab labels, because ACCOUNTS and
        UNRESOLVED were separate sections and switching would otherwise hide
        one from the other -- an account list means something different once
        you know 183 sites gave no answer. One table makes that arrangement
        unnecessary and this line replaces it, with the numbers next to the
        control that acts on them rather than on a tab two widgets away.

        The unresolved count is reported LOUDER when it is being filtered out,
        not quieter. `show` refuses to let "we could not tell" read as "nobody
        was home"; a filter that silently dropped 183 rows would undo that in
        the one place someone is most likely to conclude a scan found nothing.
        """
        line = Text()
        line.append(f"{found} found", style="green" if found else "dim")
        line.append("  ·  ", style="dim")
        if unresolved and self._found_only:
            line.append(f"{unresolved} unresolved", style="yellow")
            line.append(" hidden", style="dim")
        else:
            line.append(
                f"{unresolved} unresolved", style="dim" if not unresolved else ""
            )
        self.query_one("#sites-counts", Static).update(line)

    @on(Tabs.TabActivated, "#detail-tabs")
    def _switch_section(self, event: Tabs.TabActivated) -> None:
        section = SECTION_FOR_TAB.get(event.tab.id or "")
        if section is not None:
            self.query_one("#detail-switch", ContentSwitcher).current = section
            # The filter belongs to SITES. On PROFILE it would be a control
            # that cannot do anything, which is worse than an absent one -- the
            # same rule the scan pane's anchors block follows.
            self.query_one("#sites-controls").display = section == SEC_SITES

    def on_resize(self) -> None:
        self._fit_keys()

    def _fit_keys(self) -> None:
        """Take the key off screen when the header cannot afford it.

        Hiding the box does not change the width it is measured against -- the
        header band is the full width of the detail pane either way -- so this
        settles rather than oscillating around the threshold.
        """
        width = self.query_one("#detail-head").size.width
        # Zero before the first layout. Leaving it alone rather than guessing
        # keeps the box from flickering off and on again during mount.
        if width:
            self.query_one("#detail-keys", Static).display = (
                width >= KEY_MIN_DETAIL_WIDTH
            )

    def _redraw_found_only(self) -> None:
        """Draw the filter as the same control the scan pane's toggles are.

        A `Button` with the `‹ on ›` brackets, not a checkbox or a keybinding
        alone: those brackets already mean "press to change this" on the
        settings editor and on `analysis` and `verbose`, and one visual
        language across the app is worth more than a control tuned for this
        pane alone. It reports its own state, which is the part a keybinding
        cannot do.
        """
        self.query_one("#toggle-found-only", Button).label = (
            f"found only ‹ {'on' if self._found_only else 'off'} ›"
        )

    @on(Button.Pressed, "#toggle-found-only")
    def _pressed_found_only(self) -> None:
        self.action_toggle_found_only()

    def action_toggle_found_only(self) -> None:
        """Show the sites that gave no answer, or hide them again.

        A binding as well as the button, so the footer advertises it -- the
        button is reachable only by cycling focus with Tab, which is true and
        undiscoverable, the same argument the section bindings make.
        """
        self._found_only = not self._found_only
        self._redraw_found_only()
        if self._record is not None and self._record.get("known"):
            self._show_record(self._record)

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
        self._fill_sites(accounts, unresolved)
        self._redraw_counts(found=len(accounts), unresolved=len(unresolved))
        self._seed_build_anchors(record)
        self.query_one("#detail-profile", Static).update(
            self._profile_block(record)
        )
        self._redraw_profile_actions(record)

    def _fill_sites(
        self,
        accounts: list[dict[str, Any]],
        unresolved: list[dict[str, Any]],
    ) -> None:
        """One table for what was found and what could not be decided.

        These were two sections, and `show` still keeps the two lists apart for
        a reason worth restating: they answer different questions, and merging
        them is what let silence read as absence in the first place. What made
        that merge dangerous was dropping the distinction, not showing the rows
        together -- and nothing is dropped here. Every row wears the status it
        was stored with, the key above names all four, and the line above the
        table reports the unresolved count even while it is filtered out.

        Two sections cost more than they protected: you could sit on ACCOUNTS
        and never learn UNRESOLVED existed. One table with 183 unanswered rows
        in it cannot be read that way.

        Hits first, then the rest, each name-sorted -- interleaved by name, the
        handful of findings would be scattered through hundreds of rows that
        are not findings.
        """
        table = self.query_one(f"#{SEC_SITES}", DataTable)
        table.clear()
        self._row_urls.clear()

        found = status_style(QueryStatus.CLAIMED)
        rows: list[tuple[str, str, Text, Text, str]] = []
        for account in accounts:
            note = []
            if account.get("confidence") and account["confidence"] != "Confirmed":
                note.append(str(account["confidence"]))
            # A hit found without a browser is weaker evidence than one found
            # with it, and this is the only place months later that says so --
            # the same annotation `show` makes, for the same reason.
            if account.get("transport") == "http":
                note.append("no browser")
            rows.append((
                found.glyph,
                found.style,
                Text(str(account["site_name"]), overflow="ellipsis", no_wrap=True),
                Text.assemble(
                    (str(account["url"]), ""),
                    (f"  [{'; '.join(note)}]" if note else "", "dim"),
                ),
                str(account["url"]),
            ))

        if not self._found_only:
            for entry in unresolved:
                # The stored status, not a prefix match on the sentence written
                # from it. Reading the symbol back out of its own explanation
                # meant anything not starting with "blocked" was drawn as
                # inconclusive -- so a username the site's own rules reject,
                # which `show` reports as "username format rejected" and which
                # has its own ✕, arrived wearing the symbol for "we could not
                # tell". That is exactly the conflation this list exists to
                # prevent, and the key would name it wrong just as confidently.
                style = status_style(status_from_name(entry.get("status")))
                detail = entry["reason"]
                if entry.get("transport") == "http":
                    # Often the whole explanation for an inconclusive result, so
                    # it goes before the symptom rather than behind it.
                    detail = f"{detail}; no browser"
                if entry.get("context"):
                    detail = f"{detail}; {entry['context']}"
                rows.append((
                    style.glyph,
                    style.style,
                    Text(str(entry["site_name"]), overflow="ellipsis", no_wrap=True),
                    Text(detail, style="dim", overflow="ellipsis", no_wrap=True),
                    str(entry.get("url") or ""),
                ))

        if not rows:
            # One row rather than an empty table, so the section reads as
            # answered rather than as still loading -- and when the filter is
            # what emptied it, the row says so instead of leaving someone to
            # conclude the scan found nothing.
            if self._found_only and unresolved:
                message = (
                    f"no accounts found — {count_of(len(unresolved), 'site')} "
                    "gave no answer; press f to see them"
                )
            else:
                message = "no accounts found"
            table.add_row(Text(""), Text("—", style="dim"),
                          Text(message, style="dim italic"))
            return

        for index, (glyph, glyph_style, site, detail, url) in enumerate(rows):
            key = str(index)
            self._row_urls[key] = url
            table.add_row(Text(glyph, style=glyph_style), site, detail, key=key)

    @on(DataTable.RowSelected, f"#{SEC_SITES}")
    def _open_row(self, event: DataTable.RowSelected) -> None:
        """Enter on a row opens the site it is about.

        This is where the action matters most: the live feed scrolls past, but
        this list is what someone comes back to days later, and retyping a URL
        off a terminal is exactly the friction the pane exists to remove. It
        works on an unresolved row too -- going to look for yourself is the
        obvious next move from a row that says nothing could be decided.
        """
        url = self._row_urls.get(event.row_key.value or "")
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
