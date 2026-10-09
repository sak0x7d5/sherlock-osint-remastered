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
glossary sits on the filter line directly above the column, on SITES only.

It was a bordered box in the header first, on every section -- including the
two that draw no symbol -- and it cost the header four rows. Now it shares the
filter line when there is room, takes a line of its own under it when there is
not, and goes only on a very narrow pane, because the counts on that line are
the part that must not clip.

Not beside the tabs, which is where the spare width looks like it is. Measured,
that space is not spare: `Tabs` is a scrolling strip, so squeezed it drops tabs
rather than wrapping or ellipsizing them, silently and from the left.

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

**Neither write is a mark on the list.** Deleting was a `✕` on the row under
the pointer first, and a `DataTable` cannot hold a control: the row cursor
paints every cell of its row. Then it was a permanent red button in the record
header -- the loudest thing on screen, on every section, and the first Tab stop
after the list. It now lives last in the `⋯ Actions` menu beside the name,
apart from the safe actions and red only there, behind a confirmation. Building
lives on PROFILE, directly under the card that says what a build would do.

**Focus always lands somewhere visible.** Switching section moves focus into
the section, and the narrow layout moves it off the list it hides. A section
that is hidden takes its focused widget with it, and Textual then focuses
nothing: every key on this pane went unheard on PROFILE until that was fixed.

**A rebuild keeps the profile's own anchors.** `--ai-synthesize-only` takes them
from the command line and never from the profile it overwrites, so rebuilding
without re-typing `--anchor` abandons the identity resolution rather than
recomputing it -- measured on real data as 4 fields / 7 values / 2 anchors
becoming 11 fields / 73 values / 0. Here they are seeded from the stored profile
and shown above the button, so the safe default is also the visible one.
"""

from __future__ import annotations

import webbrowser
from datetime import datetime
from io import StringIO
from pathlib import Path
from time import perf_counter
from typing import Any, ClassVar

from rich.console import Console, Group
from rich.text import Text
from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Grid, Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
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
SEC_EXTRACTIONS = "sec-extractions"
SEC_PROFILE = "sec-profile"

TAB_SITES = "tab-sites"
TAB_EXTRACTIONS = "tab-extractions"
TAB_PROFILE = "tab-profile"

# Section order is the order of a review: what the scan found, what the model
# made of it, and the profile that merges that. So EXTRACTIONS sits between the
# raw results and the synthesis it feeds, which is also where it falls in the
# pipeline.
SECTION_FOR_TAB = {
    TAB_SITES: SEC_SITES,
    TAB_EXTRACTIONS: SEC_EXTRACTIONS,
    TAB_PROFILE: SEC_PROFILE,
}

# Set on `#profile-actions` while the only thing on offer is a trip to the scan
# tab. A class rather than per-widget style writes, because what changes is the
# WEIGHT of the whole row -- layout, chrome, alignment -- and three of those
# live in the stylesheet already. See the rules it drives in `theme.py`.
NO_EVIDENCE = "-no-evidence"

from sherlock_project.database import (
    SherlockDB,
    SiteExtractionRecord,
    StoredUsernameListing,
    default_database_path,
)
from sherlock_project.profile_synthesis import IdentityAnchor
from sherlock_project.result import QueryStatus
from sherlock_project.tui.anchor_screen import AnchorScreen
from sherlock_project.tui.confirm_screen import ConfirmScreen
from sherlock_project.tui.extraction_view import (
    current_contract_hash,
    extraction_detail,
    extraction_summary,
    facts_cell,
    sort_extractions,
)
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
KEY_STATUSES: tuple[QueryStatus, ...] = (
    QueryStatus.CLAIMED,
    QueryStatus.UNKNOWN,
    QueryStatus.WAF,
    QueryStatus.ILLEGAL,
)

# Detail-pane widths for the status key. At KEY_INLINE_WIDTH and above it
# shares one line with the filter chip and the counts; below that it drops to a
# line of its own under them -- a row of findings is a fair price for knowing
# what the symbols mean -- and below KEY_MIN_DETAIL_WIDTH it goes, because the
# counts are the part of that band that must survive and the key is a glossary.
KEY_INLINE_WIDTH = 100
KEY_MIN_DETAIL_WIDTH = 46

# Pane width below which the username list stops being a column and becomes a
# picker over the detail (ctrl+l). At 80 columns the 37-cell list took almost
# half the screen and left the record a link column three characters wide.
NARROW_WIDTH = 100

def _unless_torn_down(method):
    """Skip a late callback whose widgets are already gone.

    Several things here finish AFTER the event that started them: a database
    read in a worker, a measurement queued with `call_after_refresh`, a timer
    tick. If the app closes in between, the pane's children are removed first
    and the callback's first `query_one` raises `NoMatches` -- inside the app,
    on its way out, which is how a test that never touched this tab failed on
    Windows CI. There is nothing left to draw, so doing nothing is correct.
    """

    def guarded(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except NoMatches:
            if self.is_attached and self.children:
                # Still mounted: a missing widget is a real bug, not teardown.
                raise
            return None

    guarded.__name__ = method.__name__
    guarded.__doc__ = method.__doc__
    return guarded


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
    #
    # Section-scoped keys are filtered by `check_action`, so the footer lists
    # only what does something in the section on screen. A footer that offered
    # `found only` on PROFILE and `notes` on SITES taught people that keys here
    # do nothing.
    BINDINGS: ClassVar = [
        Binding("f", "toggle_found_only", "found only"),
        Binding("b", "build_profile", "build"),
        Binding("a", "edit_anchors", "anchors"),
        Binding("s", "toggle_sources", "sources"),
        Binding("v", "toggle_notes", "notes"),
        # ctrl+arrows as well, for terminals whose alt+arrow never arrives as
        # one: macOS Terminal.app sends Option+arrow as ESC f / ESC b, the
        # readline word-jump, and Textual decodes those as ctrl+right/left.
        # Nothing on this pane takes ctrl+arrows -- it has no text field, and
        # DataTable binds only the plain arrows.
        Binding("alt+right,ctrl+right", "next_section", "section"),
        Binding("alt+left,ctrl+left", "prev_section", "prev section", show=False),
        Binding("m", "record_actions", "actions"),
        Binding("ctrl+l", "toggle_picker", "usernames"),
        Binding("ctrl+e", "export", "export", show=False),
        Binding("ctrl+r", "reload", "refresh", show=False),
        Binding("delete", "delete_username", "delete…", show=False),
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
        # The username at the top of the list when it was last loaded, so a
        # reload can tell a freshly scanned name from a list that only shuffled.
        self._last_top: str | None = None
        # Which section is showing, for `check_action` -- the footer offers a
        # section's keys only while that section is the one on screen.
        self._section = SEC_SITES
        # Below NARROW_WIDTH the username list is a picker over the detail
        # rather than a column beside it. See `_fit_width`.
        self._narrow = False
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
        # Site name -> its stored extraction, for the whole selected username.
        # Loaded once per username with the rest of the detail rather than
        # per-row on demand: the facts column needs every row's count to draw at
        # all, so a query per row would be one round trip per account.
        self._extractions: dict[str, SiteExtractionRecord] = {}
        # The same records in the order the panel lists them, so a row index maps
        # straight to a record. Kept beside the dict rather than re-sorting on
        # every cursor move -- the cursor moves on every arrow key, and sorting
        # 149 records per keypress to answer "which one is row 4" is work for an
        # answer that has not changed.
        self._extraction_order: list[SiteExtractionRecord] = []
        # Whether ANY site for this username has been analysed. Gates the detail
        # so a per-site "not analysed" panel cannot replace the whole-username
        # explanation -- see `_fill_extractions`.
        self._any_analysed = False
        # Live state for a rebuild in progress.
        self._build_reporter: TuiReporter | None = None
        self._build_started = 0.0
        self._build_tick = 0
        self._building = False

    def compose(self) -> ComposeResult:
        yield Static("STORED USERNAMES", classes="pane-title")
        with Grid(id="results-body"):
            # `cursor_foreground_priority` so the hit count keeps its green
            # on the SELECTED row. `DataTable` otherwise repaints every cell of
            # that row in the cursor's own colour, and green is the only thing
            # in this list saying an account was found.
            yield DataTable(
                id="username-list",
                cursor_type="row",
                cursor_foreground_priority="renderable",
            )
            with Vertical(id="result-detail"):
                # Identity left, key right. The right half of this band was
                # empty at every width -- a username and a timestamp do not
                # fill 90 cells -- so the glossary goes in the space the header
                # was already paying for rather than in a row of its own.
                with Grid(id="detail-head"):
                    # Who this is and what the scan came to, on two lines that
                    # never wrap. Always visible, whichever section is showing.
                    yield Static(id="detail-header")
                    # Everything that can be DONE to the record, behind one
                    # quiet chip. Delete used to be a permanent red button
                    # here: the most saturated thing on screen, the first Tab
                    # stop after the list, and on every section. It now lives
                    # last in this menu, and the chip is not a Tab stop -- `m`
                    # opens it from the keyboard, and the footer says so.
                    yield Button("⋯ Actions", id="record-actions", classes="chip")
                yield Tabs(
                    Tab("SITES", id=TAB_SITES),
                    Tab("EXTRACTIONS", id=TAB_EXTRACTIONS),
                    Tab("PROFILE", id=TAB_PROFILE),
                    id="detail-tabs",
                )
                # The filter, what it is hiding, and the key to the symbols in
                # the table under it -- one line, only on SITES. The key used to
                # be a bordered box in the header on every section, including
                # the two that draw no status symbol.
                with Grid(id="sites-controls"):
                    yield Button(id="toggle-found-only", classes="chip")
                    yield Static(id="sites-counts")
                    yield Static(id="sites-key")
                # Each section owns its own scrolling, and only one is mounted
                # visible at a time -- which is the whole fix for the double
                # scrollbar.
                with ContentSwitcher(initial=SEC_SITES, id="detail-switch"):
                    yield DataTable(id=SEC_SITES, cursor_type="row")
                    # The one section that is itself master-detail, because
                    # reviewing extractions is a sweep: the cursor moves down
                    # the list and the reading beside it changes, rather than
                    # every site costing an open and a close.
                    #
                    # Two scroll regions live here, side by side, and that does
                    # NOT reintroduce the double-scrollbar defect. That was a
                    # `DataTable` nested INSIDE a scrolling column -- two bars in
                    # one column, one wrapping the other. These are independent
                    # regions in separate grid cells, exactly like the username
                    # list and this detail pane one level up.
                    with Grid(id=SEC_EXTRACTIONS):
                        with Vertical(id="extraction-list-col"):
                            yield DataTable(
                                id="extraction-list",
                                cursor_type="row",
                            )
                            # The distribution, under the list it describes. Any
                            # one extraction judges a page; these three numbers
                            # judge the model, which is the question that makes
                            # someone change it.
                            yield Static(id="extraction-summary")
                        with VerticalScroll(id="extraction-detail-col"):
                            yield Static(id="extraction-detail")
                    with VerticalScroll(id=SEC_PROFILE):
                        # The actions come FIRST. With a profile on screen they
                        # are a two-line status bar with Rebuild in it; below a
                        # long profile, Rebuild needed a scroll to find. With
                        # no profile they are the whole section: a card that
                        # reads evidence, anchors, result, then the button
                        # directly under them -- not right-aligned forty cells
                        # away from the sentence it acts on.
                        with Vertical(id="profile-actions"):
                            yield Static(id="profile-anchor-line")
                            # Progress reports HERE, not in the scan tab's
                            # ACTIVITY log: that is a different tab, and being
                            # told to go and watch somewhere else is not
                            # feedback. An anchored rebuild loads the model,
                            # which has been measured at 187s cold, so silence
                            # is the one thing this must not do.
                            yield Static(id="profile-status")
                            with Horizontal(id="profile-buttons"):
                                yield Button(
                                    "Build profile",
                                    variant="primary",
                                    id="profile-build",
                                )
                                yield Button(
                                    "Anchors…", id="profile-anchors", classes="chip"
                                )
                        yield Static(id="detail-profile")

    def on_mount(self) -> None:
        table = self.query_one("#username-list", DataTable)
        # Widths chosen to fit the fixed column the stylesheet gives this list,
        # padding and scrollbar included. At their previous size the last column
        # was clipped to "si" and its number could not be read.
        table.add_column("username", key="username", width=15)
        # "found" before "sites": the hit count is what someone is scanning the
        # list for, and the total is context for it. Reversed, the eye lands on
        # the larger, less interesting number first on every row.
        table.add_column("found", key="found", width=5)
        # "checked", not "sites": beside "found" the bare noun read as "sites
        # it was found on", which is the other number.
        table.add_column("checked", key="sites", width=7)

        sites = self.query_one(f"#{SEC_SITES}", DataTable)
        sites.add_column("", key="mark", width=2)
        sites.add_column("site", key="site", width=18)
        # What Pass 1 made of this site, before anyone opens anything. This is
        # the column that answers "is this model worth keeping": seeing that 8
        # of 149 sites yielded facts, and WHICH 8, is the judgement the ANALYSIS
        # counters can only report in aggregate. Narrow and before the URL, so
        # the URL keeps taking everything left over.
        sites.add_column("facts", key="facts", width=9)
        # One column for two kinds of answer: the URL where a hit was found,
        # the reason nothing was decided otherwise. Neither "url" nor "why" is
        # true of the other half, so the header names what the column IS rather
        # than what half of it happens to hold.
        sites.add_column("detail", key="detail")

        # Hidden until a record is on screen: a menu of things to do to a
        # record has nothing to act on under "Nothing scanned yet".
        actions = self.query_one("#record-actions", Button)
        actions.display = False
        # Mouse-reachable, keyboard-reachable through `m` -- but never a Tab
        # stop. Tab from the list now goes to the content, not to a menu whose
        # last item erases the record.
        actions.can_focus = False
        actions.tooltip = "Export, re-scan, copy links or delete  (m)"
        # The section strip is switched with alt+arrows or the mouse. As a Tab
        # stop it took focus invisibly -- its focus style is deliberately
        # suppressed -- so the next key seemed to go nowhere.
        self.query_one("#detail-tabs", Tabs).can_focus = False

        # Drawn once. The key is the app's vocabulary, not this record's, so
        # nothing that happens to the data can change it. One line, beside the
        # filter, on the one section whose table draws these symbols.
        self.query_one("#sites-key", Static).update(
            status_key(KEY_STATUSES, columns=len(KEY_STATUSES))
        )
        self._redraw_found_only()
        self._fit_keys()

        # Facts BEFORE site, which is the reverse of every other table here and
        # is deliberate: this list is sorted by that number, so the column the
        # order is built on is the one the eye should land on first. In the
        # account list the site is what you are looking up and the count is an
        # annotation; here the count is the subject.
        extractions = self.query_one("#extraction-list", DataTable)
        extractions.add_column("facts", key="facts", width=5)
        # 15, and the stylesheet's 28-cell column is measured against these two
        # plus DataTable's own per-cell padding. Widening either without widening
        # the column there gives this table a horizontal scrollbar.
        extractions.add_column("site", key="site", width=15)

        # Static, so it is hung once here rather than on every redraw: unlike
        # the Build button's, this text does not depend on what the selected
        # username has. Anchors are the least self-explanatory control in the
        # app -- the word names the mechanism, not the thing you would type.
        self.query_one("#profile-anchors", Button).tooltip = (
            "Anchors\n\n"
            "Facts you already know about the target — a real name, a city, an "
            "employer. Pass 2 resolves the profile against them instead of "
            "merging every name every site showed, which is the difference "
            "between one person's profile and everyone who shares the username."
        )

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

    @_unless_torn_down
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
        # Which row to open, in order of authority:
        #   1. one something asked for by name (the scan pane's "view results");
        #   2. a username that has just risen to the top -- the list is sorted
        #      most-recent-first, so a new top is a scan that just finished,
        #      and it is the one someone came here to read;
        #   3. otherwise the row that was open before the reload.
        # The third rule is the fix. This tab reloads on every visit, and it
        # used to land on row 0 each time, so checking the scan tab and coming
        # back lost your place in the list.
        names = [listing.username for listing in self._listings]
        wanted = self._pending_selection
        self._pending_selection = None
        top_changed = bool(names) and names[0] != self._last_top
        self._last_top = names[0] if names else None
        if wanted not in names:
            wanted = names[0] if top_changed else self._selected
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

        Deliberately the same payload `sherlock-rm show --json` produces, from the
        same function, so a file written here and one written there are the same
        file. `ensure_ascii` is on for the reason that command documents:
        profiles carry names outside cp1252 and Windows encodes redirected
        output with it.
        """
        if self._record is None or not self._record.get("known"):
            self.notify("Nothing to export.", severity="warning")
            return
        from sherlock_project.show import _as_json

        # Never over an earlier export. Two exports of one username are two
        # snapshots of an investigation at two moments, and the second one
        # silently replacing the first lost the earlier state with no warning.
        # The timestamp also sorts them in the order they were taken.
        stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
        target = Path.cwd() / f"{self._record['username']}-{stamp}.json"
        suffix = 2
        while target.exists():
            target = Path.cwd() / f"{self._record['username']}-{stamp}-{suffix}.json"
            suffix += 1
        try:
            target.write_text(_as_json([self._record]), encoding="utf-8")
        except OSError as error:
            self.notify(f"Could not write {target}: {error}", severity="error")
            return
        self.notify(f"Wrote {target}")

    def action_delete_username(self) -> None:
        """Erase the open username, after confirming. `del`, and the button.

        Reads the selection rather than the loaded record, so it works in the
        moment between picking a row and its detail arriving -- the detail is a
        separate database read, and a key that does nothing for a beat looks
        broken.
        """
        if self._selected is None:
            self.notify("Nothing selected to delete.", severity="warning")
            return
        self._confirm_delete(self._selected)

    @on(Button.Pressed, "#record-actions")
    def _actions_pressed(self) -> None:
        self.action_record_actions()

    def action_record_actions(self) -> None:
        """Open the menu of things that can be done to the open record.

        Ordered by risk, safest first, with Delete last and set apart -- the
        same rule a confirmation dialog follows for its buttons. Each item is
        also a key, and the menu names it, so the menu teaches the shortcuts
        rather than standing in for them.
        """
        from sherlock_project.tui.record_actions import RecordActionsScreen

        username = self._selected
        if username is None:
            return
        found = [
            str(account["url"])
            for account in (self._record or {}).get("accounts") or []
            if account.get("url")
        ] if self._record and self._record.get("username") == username else []

        def chosen(choice: str | None) -> None:
            if choice == "export":
                self.action_export()
            elif choice == "rescan":
                self.post_message(self.Rescan(username))
            elif choice == "copy":
                self.app.copy_to_clipboard("\n".join(found))
                self.notify(f"Copied {count_of(len(found), 'link')}.")
            elif choice == "profile":
                self._show_section(TAB_PROFILE)
            elif choice == "delete":
                self._confirm_delete(username)

        self.app.push_screen(
            RecordActionsScreen(username, links=len(found)), chosen
        )

    class Rescan(Message):
        """Asked to scan this username again. The app owns which tab scans."""

        def __init__(self, username: str) -> None:
            super().__init__()
            self.username = username

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

        The figures come from the LISTING rather than from the loaded detail,
        so the question can be asked the moment a row is picked. The detail is
        a second database read; waiting on it would mean the button and the key
        do nothing for a beat after every change of selection, and a dialog
        that counted a username it was no longer naming would be worse than not
        asking at all.
        """
        listing = next(
            (item for item in self._listings if item.username == username), None
        )
        if listing is None:
            self.notify("Nothing selected to delete.", severity="warning")
            return

        # Consequence first, as a list of what goes, then permanence. The name
        # is in the title and on the button, so the body does not repeat it --
        # the old body opened by asking the title's question a second time.
        detail = Text()
        detail.append("This removes everything stored for this username:\n")
        detail.append(f"  {count_of(listing.total_sites, 'site result')}, including ")
        detail.append(
            f"{count_of(listing.claimed_sites, 'found account')}\n", style="bold"
        )
        # Stored page text and Pass 1 extractions live on the result rows, so
        # they go with them -- worth saying, because they are the expensive part.
        detail.append("  the stored pages and their AI extractions")
        if listing.has_profile:
            detail.append("\n  the AI profile")
        detail.append(
            "\n\nThis cannot be undone. Getting it back means scanning again.",
            style="dim",
        )

        def erase(confirmed: bool | None) -> None:
            if confirmed:
                self._erase(username)

        self.app.push_screen(
            ConfirmScreen(
                f"Delete {username}?",
                detail,
                # The verb alone. The username is the title's job: in the
                # button it made the button as wide as the name, and a long
                # name turned the dialog's most dangerous control into its
                # widest. The title wraps; a button cannot.
                confirm_label="Delete",
                cancel_label="Keep it",
                danger=True,
            ),
            erase,
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
            # On the same connection as the record it annotates. A second
            # connection opened just for this would race schema init against the
            # first on a fresh database -- the bug `_ensure_column` already had
            # to be made idempotent for.
            extractions = await db.get_site_extractions(username)
        finally:
            await db.close()
        self._record = record
        # Keyed by site name because that is what the accounts rows carry.
        # `_collect` returns display records, not site ids, and adding ids to it
        # would change a shape `show --json` also renders.
        self._extractions = {
            entry.site_name: entry for entry in extractions
        }
        self._show_record(record)

    @_unless_torn_down
    def _set_detail(self, renderable: Any) -> None:
        """Show a bare message in place of a record."""
        self.query_one("#detail-header", Static).update(renderable)
        # An empty database, or a username with nothing stored: the header is
        # a sentence now, not an identity. Everything that acts on a record --
        # the actions menu, the sections and the filter -- goes with it, rather
        # than sitting under the sentence with nothing to act on.
        self._show_record_chrome(False)
        self.query_one(f"#{SEC_SITES}", DataTable).clear()
        self.query_one("#detail-profile", Static).update("")
        self._row_urls.clear()
        # Cleared with the rows they annotate. Left behind, a later username's
        # GitHub row would show the previous username's extraction -- the rows
        # are keyed by site name, which two usernames routinely share.
        self._extractions.clear()
        self._extraction_order.clear()
        self._any_analysed = False
        self.query_one("#extraction-list", DataTable).clear()
        self.query_one("#extraction-summary", Static).update("")
        self.query_one("#extraction-detail", Static).update("")
        self._redraw_counts(found=0, unresolved=0)

    def _show_record_chrome(self, shown: bool) -> None:
        """Show or hide everything that only means something with a record open."""
        self.query_one("#record-actions", Button).display = shown
        self.query_one("#detail-tabs", Tabs).display = shown
        self.query_one("#detail-switch", ContentSwitcher).display = shown
        self.query_one("#sites-controls").display = shown and (
            self._section == SEC_SITES
        )
        self.refresh_bindings()

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

        And it is the part of this line that must survive a narrow terminal.
        At 80 columns it was clipped clean off the end -- "6 found ·" and then
        nothing -- so the hidden count now comes FIRST when there is one, and
        the line wraps rather than clipping.
        """
        line = Text()
        if unresolved and self._found_only:
            line.append(f"{unresolved} unresolved hidden", style="yellow")
            line.append("  ·  ", style="dim")
            line.append(f"{found} found", style="green" if found else "dim")
        else:
            line.append(f"{found} found", style="green" if found else "dim")
            line.append("  ·  ", style="dim")
            line.append(
                f"{unresolved} unresolved", style="dim" if not unresolved else ""
            )
        self.query_one("#sites-counts", Static).update(line)

        # Extractions counts what the model FOUND SOMETHING on, not how many
        # sites are listed -- the line above already carries the site counts.
        # The useful number here is the yield, because `EXTRACTIONS 8` beside
        # `147 found` is the quality signal legible without opening the
        # section at all.
        with_facts = sum(
            1 for record in self._extractions.values() if record.fact_count
        )
        tabs = self.query_one("#detail-tabs", Tabs)
        tabs.query_one(f"#{TAB_EXTRACTIONS}", Tab).label = (
            f"EXTRACTIONS {with_facts}" if with_facts else "EXTRACTIONS"
        )
        # The underline is measured against the label it sits under, and a
        # label that changes width after layout left it drawn under the OLD
        # extent -- "3 PROFI" underlined while PROFILE was the section open.
        # Re-measured once the new label has been laid out.
        self.call_after_refresh(self._rehighlight_tabs)

    @_unless_torn_down
    def _rehighlight_tabs(self) -> None:
        self.query_one("#detail-tabs", Tabs)._highlight_active(animate=False)

    @on(Tabs.TabActivated, "#detail-tabs")
    def _switch_section(self, event: Tabs.TabActivated) -> None:
        section = SECTION_FOR_TAB.get(event.tab.id or "")
        if section is None:
            return
        self._section = section
        self.query_one("#detail-switch", ContentSwitcher).current = section
        # The filter only means something on SITES; a control that cannot do
        # anything is worse than an absent one -- the same rule the scan pane's
        # anchors block follows.
        self.query_one("#sites-controls").display = section == SEC_SITES
        # The footer lists each section's own keys, so it has to be re-asked.
        self.refresh_bindings()
        # FOCUS MOVES INTO THE SECTION, every time. Hiding a section hides the
        # widget that had focus in it, and Textual then focuses nothing at all
        # -- which on PROFILE meant every key on this pane went unheard, Tab
        # included, and the only way out was the mouse or another tab. `v` and
        # `s` exist only for PROFILE and could never be pressed there.
        #
        # Only when focus is already in this pane, or nowhere while the pane
        # is on screen. The strip announces its first section while the app
        # is still mounting, on a tab nobody is looking at -- moving focus then
        # took it out of the username field on SCAN, and the first keystrokes
        # of a session went nowhere.
        focused = self.app.focused
        if focused is None:
            if not self.region.area:
                return
        elif self not in focused.ancestors_with_self:
            return
        self._focus_section()

    @_unless_torn_down
    def focus_default(self) -> None:
        """Where focus lands when this tab is opened: the list, or -- when the
        list is a hidden picker -- the open section, so keys are heard at once."""
        if self._narrow and not self.has_class("-picking"):
            self._focus_section()
        else:
            self.query_one("#username-list", DataTable).focus()

    @_unless_torn_down
    def _focus_section(self) -> None:
        target = {
            SEC_SITES: f"#{SEC_SITES}",
            # The extraction list holds focus because the arrow keys ARE that
            # panel -- the detail follows the cursor.
            SEC_EXTRACTIONS: "#extraction-list",
            # The scroller itself, so the profile scrolls with the arrows and
            # the section's keys reach this pane.
            SEC_PROFILE: f"#{SEC_PROFILE}",
        }[self._section]
        self.query_one(target).focus()

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """Offer a key only where it does something.

        False takes the binding out of the footer and stops it firing, which
        is what "does nothing here" should look like.
        """
        record = self._record if self._record and self._record.get("known") else None
        if action == "toggle_found_only":
            return record is not None and self._section == SEC_SITES
        if action in ("toggle_sources", "toggle_notes"):
            return (
                record is not None
                and self._section == SEC_PROFILE
                and record.get("profile") is not None
            )
        if action in ("build_profile", "edit_anchors"):
            return (
                record is not None
                and self._section == SEC_PROFILE
                and bool(self._extraction_count(record))
                and not self._building
            )
        if action in ("record_actions", "export", "delete_username"):
            return self._selected is not None
        if action in ("next_section", "prev_section"):
            return record is not None
        if action == "toggle_picker":
            return self._narrow
        return True

    def on_resize(self) -> None:
        self._fit_width()
        self._fit_keys()

    @_unless_torn_down
    def _fit_keys(self) -> None:
        """Put the key beside the counts, under them, or nowhere.

        The counts are the part of that band that must not clip, so the key is
        the one to move -- it is a glossary, and the record is what it explains.
        """
        width = self.query_one("#result-detail").size.width
        # Zero before the first layout. Leaving it alone rather than guessing
        # keeps the key from flickering off and on again during mount.
        if not width:
            return
        self.query_one("#sites-key", Static).display = width >= KEY_MIN_DETAIL_WIDTH
        self.query_one("#sites-controls").set_class(
            width < KEY_INLINE_WIDTH, "-stacked"
        )

    @_unless_torn_down
    def _fit_width(self) -> None:
        """Turn the username list into a picker below NARROW_WIDTH.

        The list is a fixed 37 cells, which is right beside a wide detail and
        wrong at 80 columns: there it took almost half the screen, the link
        column shrank to "htt", and the extraction reading pane to eight cells.
        Narrow, the record gets the full width and the list is one key away
        (ctrl+l), drawn over the detail rather than beside it.
        """
        width = self.size.width
        if not width:
            return
        narrow = width < NARROW_WIDTH
        if narrow == self._narrow:
            return
        self._narrow = narrow
        self.set_class(narrow, "-narrow")
        if not narrow:
            self.remove_class("-picking")
        self._redraw_header()
        self.refresh_bindings()
        # The detail is a different width once the list has moved, and the
        # key is measured against the detail -- so measure again after the
        # new layout lands, not against the one it replaced.
        self.call_after_refresh(self._fit_keys)
        # Going narrow hides the list. If it had focus, focus has to go
        # somewhere visible, or every key on the tab goes unheard.
        focused = self.app.focused
        username_list = self.query_one("#username-list", DataTable)
        if (
            narrow
            and (focused is None or focused is username_list)
            and self.region.area
        ):
            self.call_after_refresh(self._focus_section)

    def action_toggle_picker(self) -> None:
        """Show the username list over the detail, or put it away again."""
        if not self._narrow:
            return
        picking = not self.has_class("-picking")
        self.set_class(picking, "-picking")
        if picking:
            self.query_one("#username-list", DataTable).focus()
        else:
            self._focus_section()

    @on(DataTable.RowSelected, "#username-list")
    def _picked(self) -> None:
        """Enter on a username closes the picker onto it."""
        if self.has_class("-picking"):
            self.remove_class("-picking")
            self._focus_section()

    def _redraw_found_only(self) -> None:
        """Draw the filter as a checkbox chip, like every other toggle here.

        A ticked box, not `‹ on ›`. Those brackets mean "step through values"
        on the settings screen, where left and right really do step; on a
        toggle they promised a spinner and delivered a switch. ☑ and ☐ say
        "this is on" in the shape every checkbox has, without reading a word.
        """
        mark = "☑" if self._found_only else "☐"
        self.query_one("#toggle-found-only", Button).label = f"{mark} found only"

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

    def _show_section(self, tab_id: str) -> None:
        self.query_one("#detail-tabs", Tabs).active = tab_id

    # -- rendering ----------------------------------------------------------

    @_unless_torn_down
    def _show_record(self, record: dict[str, Any]) -> None:
        if not record.get("known"):
            self._set_detail(
                Text("Nothing stored for this username.", style="dim italic")
            )
            return

        unresolved = record.get("unresolved") or []
        accounts = record["accounts"]

        self._redraw_header()
        self._show_record_chrome(True)
        self._fill_sites(accounts, unresolved)
        self._fill_extractions()
        self._redraw_counts(found=len(accounts), unresolved=len(unresolved))
        self._seed_build_anchors(record)
        self.query_one("#detail-profile", Static).update(
            self._profile_block(record)
        )
        self._redraw_profile_actions(record)

    @_unless_torn_down
    def _redraw_header(self) -> None:
        """Two lines that never wrap: who, then what the scan came to.

        It was a name, a timestamp that wrapped onto a third line, a red Delete
        button and a four-row bordered key -- five rows before any data on
        every section, which at 80x24 left about twelve for the table. The
        counts come first on the second line because they are what the record
        IS; when it was scanned is context for them.
        """
        record = self._record
        if record is None or not record.get("known"):
            return
        accounts = record.get("accounts") or []
        unresolved = record.get("unresolved") or []
        name = Text(no_wrap=True, overflow="ellipsis")
        if self._narrow:
            # Narrow, the list is a picker, so the header is where you learn
            # there is one and which entry you are on.
            names = [listing.username for listing in self._listings]
            position = (
                f"  {names.index(record['username']) + 1} of {len(names)}"
                if record["username"] in names
                else ""
            )
            name.append("▾ ", style="dim")
            name.append(str(record["username"]), style="bold cyan")
            name.append(f"{position} · ^l list", style="dim")
        else:
            name.append(str(record["username"]), style="bold cyan")
        facts = Text(no_wrap=True, overflow="ellipsis")
        facts.append(
            f"{len(accounts)} found", style="green" if accounts else "dim"
        )
        if unresolved:
            facts.append(" · ", style="dim")
            facts.append(f"{len(unresolved)} unresolved", style="yellow")
        facts.append(
            f" · {record['sites_checked']} checked"
            f" · scanned {record['last_scanned_at']}",
            style="dim",
        )
        self.query_one("#detail-header", Static).update(Group(name, facts))

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
        rows: list[tuple[str, str, Text, Text, str, str]] = []
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
                str(account["site_name"]),
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
                    str(entry["site_name"]),
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
            table.add_row(Text(""), Text("—", style="dim"), Text(""),
                          Text(message, style="dim italic"))
            return

        for index, (glyph, glyph_style, site, detail, url, name) in enumerate(rows):
            key = str(index)
            self._row_urls[key] = url
            table.add_row(
                Text(glyph, style=glyph_style),
                site,
                # The same renderer the extraction panel's own list uses, so one
                # site cannot report a different count depending on which list
                # you are looking at.
                facts_cell(self._extractions.get(name)),
                detail,
                key=key,
            )

    def _fill_extractions(self) -> None:
        """Draw the extraction list, its summary, and the first detail.

        Ordered by `sort_extractions` rather than by site name -- see there for
        why, but in short: this list exists to be read from both ends, and
        alphabetical order buries both in the middle.
        """
        table = self.query_one("#extraction-list", DataTable)
        table.clear()
        self._extraction_order = sort_extractions(self._extractions.values())

        summary = self.query_one("#extraction-summary", Static)
        detail = self.query_one("#extraction-detail", Static)

        if not self._extraction_order:
            # No claimed sites at all, so there is nothing Pass 1 could ever
            # have run on. Distinct from the case below, where there are sites
            # and none was analysed.
            summary.update("")
            detail.update(
                Text(
                    "No accounts were found for this username, so there is "
                    "nothing to extract from.",
                    style="dim italic",
                )
            )
            return

        for index, record in enumerate(self._extraction_order):
            table.add_row(
                facts_cell(record),
                Text(record.site_name, overflow="ellipsis", no_wrap=True),
                key=str(index),
            )
        summary.update(extraction_summary(self._extraction_order))

        self._any_analysed = any(
            record.analysed for record in self._extraction_order
        )
        if not self._any_analysed:
            # Sites exist and none was analysed -- the username was scanned
            # without analysis. Say so once, here, rather than making someone
            # arrow down 149 identical "not analysed" panels to work it out.
            #
            # The flag is what KEEPS this on screen. `add_row` above posts a
            # `RowHighlighted` for the first row, which Textual delivers after
            # this method returns -- so without the guard in `_show_extraction`
            # the handler overwrote this message with a per-site "not analysed"
            # panel a tick later, and the whole-username explanation was
            # unreachable.
            detail.update(
                Text(
                    "None of these sites has been analysed.\n\n"
                    "Scan this username again with analysis on to extract "
                    "from the pages already stored.",
                    style="dim italic",
                )
            )
            return

        # The most productive site, because it is first in the order and because
        # a panel that opens blank asks the reader to do work before it tells
        # them anything.
        self._show_extraction(0)

    @on(DataTable.RowHighlighted, "#extraction-list")
    def _extraction_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """The detail follows the CURSOR, not a selection.

        `RowHighlighted`, never `RowSelected`: this is the whole reason the
        panel replaced a dialog. Arrowing down the list sweeps through
        extractions, which is how a prompt edit is judged across many pages --
        requiring Enter on each one would make the panel a dialog with extra
        steps.
        """
        self._show_extraction_key(event.row_key.value)

    def _show_extraction_key(self, key: str | None) -> None:
        if key is None:
            return
        try:
            self._show_extraction(int(key))
        except ValueError:
            return

    def _show_extraction(self, index: int) -> None:
        if not self._any_analysed:
            # The whole-username "nothing analysed" message is on screen and is
            # the right answer for every row, so no row may replace it with the
            # per-site version of the same news.
            return
        if not 0 <= index < len(self._extraction_order):
            return
        self.query_one("#extraction-detail", Static).update(
            extraction_detail(
                self._extraction_order[index],
                # Worked out once per render rather than cached on the pane: it
                # reads one prompt file, and a cached value would go stale
                # exactly when someone is editing that file to see the effect.
                contract_hash=current_contract_hash(),
            )
        )

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
        # `b` and `a` leave the footer while the work runs, with the buttons.
        self.refresh_bindings()
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
            #
            # This control is a POINTER, not a commit: pressing it scans
            # nothing, it switches tabs and sets the scan up. Dressed as a
            # primary block it was indistinguishable from SCAN, which starts a
            # 680-site run, and from `Build profile`, which can block for
            # minutes on a cold model -- the app's own rule is that chrome
            # weight tracks what a control commits, which is why the scan
            # pane's toggles are flat. The class carries that rule here.
            actions.add_class(NO_EVIDENCE)
            # Two different problems wearing one label until now. Pages are
            # stored for every result whether or not analysis was on, so a
            # username scanned without it is not missing evidence -- it is
            # holding unread evidence, and reading it costs no network at all.
            # Saying "scan again" there sent people to a 680-site refetch for
            # work the stored pages already support.
            pending = int(record.get("pending_analysis") or 0)
            if pending:
                hint.update(
                    Text.assemble(
                        ("No AI evidence stored\n", "bold"),
                        (
                            (
                                f"This username was scanned without analysis, "
                                f"but {count_of(pending, 'page')} "
                                f"{'is' if pending == 1 else 'are'} stored and "
                                f"ready to read. Analysing them re-fetches "
                                f"nothing."
                            ),
                            "dim",
                        ),
                    )
                )
                build.label = f"▸ Analyse {count_of(pending, 'stored page')}"
                # The stored-pages arm gets its own hover text, not the scan
                # one: this button re-fetches nothing, and a tooltip promising a
                # scan would describe the opposite of what pressing it does.
                build.tooltip = (
                    "Analyse stored pages — no refetch\n\n"
                    "Loads this username on the SCAN tab with analysis on, "
                    "where the stored pages can be read without fetching them "
                    "again. Nothing starts until you press SCAN there."
                )
            else:
                # Genuinely nothing to work from: no confirmed accounts, or
                # their pages came back empty. Here a scan really is the fix.
                hint.update(
                    Text.assemble(
                        ("No AI evidence stored\n", "bold"),
                        (
                            (
                                "No stored page can be analysed, so the second "
                                "pass has nothing to merge. Scanning again with "
                                "analysis on collects it."
                            ),
                            "dim",
                        ),
                    )
                )
                # Named for where it goes and what it carries: pressing it opens
                # the scan tab with this username and analysis already set, so
                # the label promises the trip rather than a build that cannot
                # happen.
                build.label = "▸ Scan this username with analysis"
                build.tooltip = (
                    "Scan with analysis — no evidence stored\n\n"
                    "Loads this username on the SCAN tab with analysis on. "
                    "Nothing starts until you press SCAN there."
                )
            anchors.display = False
            return

        actions.remove_class(NO_EVIDENCE)
        actions.set_class(has_profile, "-has-profile")
        anchors.display = True
        count = len(self._build_anchors)
        anchors.label = f"Edit anchors ({count})…" if count else "+ Add anchor…"
        # Primary when building is the point of the section; a quiet chip once
        # a profile is on screen, where reading is the point and Rebuild is an
        # occasional correction. One raised amber button per view, and on a
        # profile view that is nothing.
        build.label = "Rebuild profile" if has_profile else "Build profile"
        build.variant = "default" if has_profile else "primary"
        build.set_class(has_profile, "chip")
        # Hover text says what the button IS; the card says what pressing it
        # would do RIGHT NOW -- how many sites of evidence, which anchors, and
        # the warning when rebuilding would abandon an identity. Splitting it
        # that way keeps the two from being one answer written twice: the
        # mechanism never changes, the state changes on every redraw.
        build.tooltip = (
            f"{'Rebuild' if has_profile else 'Build'} profile  (b)\n\n"
            "Runs the second AI pass, merging the facts Pass 1 already "
            "extracted into one profile. Reads stored evidence only — no site "
            "is contacted and no page is fetched again."
        )

        card = Text()
        if has_profile:
            # A status line over the profile, not a paragraph under it.
            card.append(str(record["username"]), style="bold")
            for part in (
                getattr(profile, "resolution_status", ""),
                getattr(profile, "mode", ""),
                getattr(profile, "completeness", ""),
            ):
                if part:
                    card.append(f" · {part}", style="dim")
            card.append(f" · built {record.get('profile_updated_at')}", style="dim")
            card.append("\n")
        else:
            card.append("No profile yet. ", style="bold")
            card.append(
                f"Evidence from {count_of(evidence, 'site')} is ready to merge.\n\n"
            )
            sources = [
                entry.site_name
                for entry in self._extraction_order
                if entry.fact_count
            ]
            card.append("EVIDENCE  ", style="dim")
            if sources:
                shown = " · ".join(sources[:4])
                more = f" +{len(sources) - 4}" if len(sources) > 4 else ""
                card.append(f"{shown}{more}\n")
            else:
                card.append(f"{count_of(evidence, 'site')} analysed\n")

        card.append("ANCHORS   " if not has_profile else "Anchored to ", style="dim")
        if self._build_anchors:
            card.append(
                " · ".join(
                    f"{a.field} = {a.value}" for a in self._build_anchors[:3]
                )
            )
            if count > 3:
                card.append(f" +{count - 3}", style="dim")
            card.append("\n")
        else:
            card.append("none\n", style="dim")

        if self._build_anchors:
            result = (
                "anchored resolution · uses the model · slow on a cold start"
            )
            if not has_profile:
                card.append("RESULT    ", style="dim")
                card.append(result)
        elif has_profile and getattr(profile, "mode", "") == "anchored":
            # The downgrade this pane exists to prevent. Rebuilding an anchored
            # profile with no anchors does not recompute the identity -- it
            # ABANDONS it, and the result looks bigger while being worse: a
            # resolved identity replaced by a merge of every name every site
            # showed. Measured at 4 fields/7 values becoming 11 fields/73.
            card.append(
                "This profile is anchored, but no anchors are set — rebuilding "
                "now would replace it with an unresolved merge.",
                style="yellow",
            )
        elif not has_profile:
            # The unanchored path needs no model at all: aggregate synthesis
            # merges stored extractions and never calls one. Worth saying,
            # because "build a profile" otherwise reads as a slow operation --
            # and so is the cost of skipping anchors, in the same breath.
            card.append("RESULT    ", style="dim")
            card.append("unanchored merge · instant · no model needed\n")
            card.append(
                "          Without anchors it describes anyone using this name.",
                style="yellow",
            )
        card.rstrip()
        hint.update(card)

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

        username = (self._record or {}).get("username")
        self.app.push_screen(
            AnchorScreen(self._build_anchors, username=username), adopt
        )

    def action_edit_anchors(self) -> None:
        """`a` on PROFILE: the anchors button, as a key."""
        self._edit_build_anchors()

    def action_build_profile(self) -> None:
        """`b` on PROFILE: the build button, as a key."""
        self._build_profile()

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
            failures = await run_synthesis_only(
                usernames=[username],
                force=rebuild,
                inline_anchors=list(self._build_anchors),
                reporter=reporter,
            )
        except Exception as error:
            # Broad on purpose, as everywhere else here: a failed synthesis
            # must not take the pane down with it.
            self._build_failed(error)
            return
        finally:
            self._build_reporter = None

        # A synthesis that fails does NOT raise: `synthesize_profiles` isolates
        # each username and reports it, so the absence of an exception says
        # nothing about whether a profile was built. Reading it as success is
        # how this button came to announce "Profile built" over a failure that
        # had left the old profile untouched.
        failure = failures.get(username)
        if failure is not None:
            self._build_failed(failure)
            return

        self._set_building(False)
        self.notify(f"Profile built for {username}.")
        self._select(username)

    def _build_failed(self, error: BaseException) -> None:
        """Put the reason on screen and give the controls back to retry with.

        Written into the status line rather than a toast -- a failure that
        disappears after five seconds is a failure nobody can act on.
        """
        self._set_building(False)
        self.query_one("#profile-status", Static).update(
            Text(f"Could not build profile: {error}", style="red")
        )

    def _profile_block(self, record: dict[str, Any]) -> Any:
        """The stored profile, or nothing -- the card above owns every other state.

        This used to add its own sentence to the no-profile states ("No profile
        stored. Scanning with a model configured builds one.") directly above a
        card saying evidence was ready and offering Build. Two paragraphs about
        one state, disagreeing, is what made the section read as improvised; one
        owner per state is the fix, and the card is that owner.
        """
        if record.get("profile_unreadable"):
            return Text(
                "A profile is stored but no longer matches the current "
                "format. Rebuild it from the CLI:\n"
                f"  sherlock-rm {record['username']} --ai-synthesize-only",
                style="yellow",
            )
        profile = record.get("profile")
        if profile is None:
            return Text("")
        return render_profile_text(
            profile,
            show_sources=self._show_sources,
            show_notes=self._show_notes,
            width=self._profile_width(),
        )

    def _profile_width(self) -> int:
        """Lay the profile out to the pane it is drawn in.

        It was rendered at a fixed 96 columns: wider than the pane at 120, so
        the right edge fell short of everything else, and wrapped at 80.
        Measured per render rather than per resize -- a toggle or a new record
        re-renders anyway, and chasing every resize event would re-render a
        profile nobody has asked to see again.
        """
        width = self.query_one(f"#{SEC_PROFILE}").size.width
        # Two cells for the scrollbar. Zero before the first layout.
        return max(40, width - 2) if width else PROFILE_RENDER_WIDTH

def render_profile_text(
    profile: Any,
    *,
    show_sources: bool = False,
    show_notes: bool = False,
    width: int = PROFILE_RENDER_WIDTH,
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
        width=width,
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
        # No box. The section is already a frame with the username above it;
        # the CLI's panel inside it drew a second border and a second title.
        framed=False,
    )
    return Text.from_ansi(buffer.getvalue())
