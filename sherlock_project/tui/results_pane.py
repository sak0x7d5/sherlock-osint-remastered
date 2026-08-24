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
SEC_ACCOUNTS = "sec-accounts"
SEC_UNRESOLVED = "sec-unresolved"
SEC_EXTRACTIONS = "sec-extractions"
SEC_PROFILE = "sec-profile"

TAB_ACCOUNTS = "tab-accounts"
TAB_UNRESOLVED = "tab-unresolved"
TAB_EXTRACTIONS = "tab-extractions"
TAB_PROFILE = "tab-profile"

# Section order is the order of a review: where the accounts are, what never
# answered, what the model made of them, and the profile that merges it. So
# EXTRACTIONS sits between the raw results and the synthesis it feeds, which is
# also where it falls in the pipeline.
SECTION_FOR_TAB = {
    TAB_ACCOUNTS: SEC_ACCOUNTS,
    TAB_UNRESOLVED: SEC_UNRESOLVED,
    TAB_EXTRACTIONS: SEC_EXTRACTIONS,
    TAB_PROFILE: SEC_PROFILE,
}

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
    status_style,
)

# Width the profile renderer is asked to lay out against. Fixed rather than the
# live pane width: `render_profile` is a three-column layout, and re-rendering
# it on every resize to chase the exact pane width costs a full re-render for a
# result nobody is measuring with a ruler. Wide enough that the columns are not
# compressed, narrow enough to fit a half-screen detail pane.
PROFILE_RENDER_WIDTH = 96


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
        # Row key -> site name, so `x` can find that site's extraction. Kept
        # separate from the URL map rather than parsed back out of the cell: the
        # site cell is drawn with `overflow="ellipsis"`, so on a long name the
        # displayed text is not the name.
        self._account_sites: dict[str, str] = {}
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
            yield DataTable(id="username-list", cursor_type="row")
            with Vertical(id="result-detail"):
                # Always visible, whichever section is showing: who this is and
                # when it was scanned is context for all three.
                yield Static(id="detail-header")
                yield Tabs(
                    Tab("ACCOUNTS", id=TAB_ACCOUNTS),
                    Tab("UNRESOLVED", id=TAB_UNRESOLVED),
                    Tab("EXTRACTIONS", id=TAB_EXTRACTIONS),
                    Tab("PROFILE", id=TAB_PROFILE),
                    id="detail-tabs",
                )
                # Each section owns its own scrolling, and only one is mounted
                # visible at a time -- which is the whole fix for the double
                # scrollbar.
                with ContentSwitcher(initial=SEC_ACCOUNTS, id="detail-switch"):
                    yield DataTable(id=SEC_ACCOUNTS, cursor_type="row")
                    yield DataTable(id=SEC_UNRESOLVED, cursor_type="row")
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

        accounts = self.query_one(f"#{SEC_ACCOUNTS}", DataTable)
        accounts.add_column("", key="mark", width=2)
        accounts.add_column("site", key="site", width=18)
        # What Pass 1 made of this site, before anyone opens anything. This is
        # the column that answers "is this model worth keeping": seeing that 8
        # of 149 sites yielded facts, and WHICH 8, is the judgement the ANALYSIS
        # counters can only report in aggregate. Narrow and before the URL, so
        # the URL keeps taking everything left over.
        accounts.add_column("facts", key="facts", width=9)
        accounts.add_column("url", key="url")

        unresolved = self.query_one(f"#{SEC_UNRESOLVED}", DataTable)
        unresolved.add_column("", key="mark", width=2)
        unresolved.add_column("site", key="site", width=18)
        unresolved.add_column("why", key="why")

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

    def _set_detail(self, renderable: Any) -> None:
        """Show a bare message in place of a record."""
        self.query_one("#detail-header", Static).update(renderable)
        self.query_one(f"#{SEC_ACCOUNTS}", DataTable).clear()
        self.query_one(f"#{SEC_UNRESOLVED}", DataTable).clear()
        self.query_one("#detail-profile", Static).update("")
        self._account_urls.clear()
        self._account_sites.clear()
        # Cleared with the rows they annotate. Left behind, a later username's
        # GitHub row would show the previous username's extraction -- the rows
        # are keyed by site name, which two usernames routinely share.
        self._extractions.clear()
        self._extraction_order.clear()
        self._any_analysed = False
        self.query_one("#extraction-list", DataTable).clear()
        self.query_one("#extraction-summary", Static).update("")
        self.query_one("#extraction-detail", Static).update("")
        self._set_tab_counts(accounts=0, unresolved=0)

    def _set_tab_counts(self, *, accounts: int, unresolved: int) -> None:
        """Put the counts on the tab labels.

        This is what makes switching safe. The stacked layout kept the
        unresolved count next to the account list because it changes what that
        list means -- six sites that never answered is not the same as six
        sites where nobody was home. Behind a bare tab that number would be
        invisible until someone thought to look, so it goes on the tab.
        """
        # Extractions counts what the model FOUND SOMETHING on, not how many
        # sites are listed. The list length is the account count, which the tab
        # beside it already carries; the useful number here is the yield, because
        # `EXTRACTIONS 8` next to `ACCOUNTS 149` is the quality signal legible
        # without opening the section at all.
        with_facts = sum(
            1 for record in self._extractions.values() if record.fact_count
        )
        labels = {
            TAB_ACCOUNTS: f"ACCOUNTS {accounts}" if accounts else "ACCOUNTS",
            TAB_UNRESOLVED: (
                f"UNRESOLVED {unresolved}" if unresolved else "UNRESOLVED"
            ),
            TAB_EXTRACTIONS: (
                f"EXTRACTIONS {with_facts}" if with_facts else "EXTRACTIONS"
            ),
        }
        tabs = self.query_one("#detail-tabs", Tabs)
        for tab_id, label in labels.items():
            tabs.query_one(f"#{tab_id}", Tab).label = label

    @on(Tabs.TabActivated, "#detail-tabs")
    def _switch_section(self, event: Tabs.TabActivated) -> None:
        section = SECTION_FOR_TAB.get(event.tab.id or "")
        if section is None:
            return
        self.query_one("#detail-switch", ContentSwitcher).current = section
        if section == SEC_EXTRACTIONS:
            # The extraction list has to hold focus or the arrow keys never
            # reach it, and the arrow keys ARE this panel -- the detail follows
            # the cursor, so a list that cannot be moved through is a panel that
            # only ever shows its first row.
            #
            # Only this section. The others are either a table the section
            # switch already leaves usable or, for PROFILE, a pane whose focus
            # behaviour is covered by its own tests; grabbing focus there would
            # change which control Enter hits.
            self.query_one("#extraction-list", DataTable).focus()

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
        self._fill_extractions()
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
        if not unresolved:
            table.add_row(
                Text(""),
                Text("—", style="dim"),
                Text("every site gave an answer", style="dim italic"),
            )
            return
        for entry in unresolved:
            style = status_style(
                QueryStatus.WAF
                if entry["reason"].startswith("blocked")
                else QueryStatus.UNKNOWN
            )
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

    def _fill_accounts(self, accounts: list[dict[str, Any]]) -> None:
        table = self.query_one(f"#{SEC_ACCOUNTS}", DataTable)
        table.clear()
        self._account_urls.clear()
        self._account_sites.clear()
        if not accounts:
            # One row rather than an empty table, so the section reads as
            # answered rather than as still loading.
            table.add_row(Text(""), Text("—", style="dim"), Text(""),
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
            site_name = str(account["site_name"])
            self._account_urls[key] = str(account["url"])
            self._account_sites[key] = site_name
            table.add_row(
                Text(found.glyph, style=found.style),
                Text(site_name, overflow="ellipsis", no_wrap=True),
                # The same renderer the extraction panel's own list uses, so one
                # site cannot report a different count depending on which list
                # you are looking at.
                facts_cell(self._extractions.get(site_name)),
                Text.assemble(
                    (str(account["url"]), ""),
                    (f"  [{'; '.join(note)}]" if note else "", "dim"),
                ),
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
