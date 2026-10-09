"""The design system every TUI screen draws from.

One file so the screens cannot drift apart. A colour picked inline on the scan
pane and a different one picked inline on the results pane is how a tool starts
looking assembled rather than designed, and it is invisible in review because
each choice reads as reasonable on its own.

Three rules hold the look together:

**Colour is never the only signal.** Every status carries a glyph as well as a
colour, because the distinction the scan reports -- found, absent, inconclusive,
blocked -- is exactly the one a colour-blind operator would lose, and it is the
distinction the whole tool exists to make. `show` already refuses to fold "the
site blocked us" into "the rules did not decide"; this keeps that refusal on
screen.

A glyph nobody has been taught is not a signal either, so every glyph is drawn
somewhere that also says what it means: `status_cell` prints the word beside it
on every row of the live feed, and `status_key` is that same sentence for the
places with room for a symbol and none for a word -- the two-cell mark column
of a table. Both read the glyph AND the word out of `STATUS_STYLES`, so a key
can never disagree with the column it sits above.

**Numbers line up in fixed columns.** Counts are read by comparing them to each
other, and a ragged left edge makes that a reading exercise instead of a glance.
The settings screen already learned this -- see LABEL_WIDTH there.

**Semantic tokens, never literal colours.** `$accent` and `$success` follow the
terminal's own light/dark theme; `#00ff00` does not, and turns unreadable on a
light background. Textual resolves these per theme, which is why the settings
screen uses them and this does too.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from rich.text import Text

from sherlock_project.result import QueryStatus

# Column widths, shared so the scan counters and the results summary agree.
# A label column wide enough for the longest thing it must hold -- a checkbox and
# "inconclusive", `☐ inconclusive` -- plus a space, so nothing is ever truncated
# into ambiguity. Every row shares it, checkbox or not, so the digits of the
# SITES counters and the ANALYSIS counters under them stack in one column.
STAT_LABEL_WIDTH = 15
STAT_VALUE_WIDTH = 6

# How often the screens repaint, in seconds. Not a cosmetic number: it is the
# thing that stops a fast scan from melting the UI. See ScanPane for why the
# scan writes into a buffer and a timer drains it, rather than every result
# touching a widget the moment it lands.
REDRAW_INTERVAL = 0.1


@dataclass(frozen=True, slots=True)
class StatusStyle:
    """How one scan outcome looks and what it is called on screen.

    `label` is deliberately not `str(QueryStatus)`. The enum says "Available",
    which reads as "this username is free to register" -- true, but the operator
    is asking "did I find them here", and the honest answer to that is "absent".
    Same for WAF, which is an implementation detail of the site, not a finding.
    """

    glyph: str
    style: str
    label: str
    # Whether the feed shows this kind of result before anyone touches the
    # filter. Only hits do. Everything else is kept -- every result is retained
    # now that the filter exists -- but drawing all of it by default would
    # scroll the handful of actual findings off screen within seconds of a
    # 680-site scan starting, which is what the default is protecting.
    default_visible: bool


STATUS_STYLES: dict[QueryStatus, StatusStyle] = {
    QueryStatus.CLAIMED: StatusStyle("●", "bold green", "found", default_visible=True),
    QueryStatus.AVAILABLE: StatusStyle("·", "dim", "absent", default_visible=False),
    QueryStatus.UNKNOWN: StatusStyle(
        "?", "yellow", "inconclusive", default_visible=False
    ),
    QueryStatus.WAF: StatusStyle("▲", "bold yellow", "blocked", default_visible=False),
    QueryStatus.ILLEGAL: StatusStyle(
        "✕", "dim italic", "rejected", default_visible=False
    ),
}


def default_visible_statuses() -> set[QueryStatus]:
    """The statuses the feed draws before the filter is touched."""
    return {
        status
        for status, style in STATUS_STYLES.items()
        if style.default_visible
    }

# The order counters are listed in, everywhere. Fixed rather than sorted by
# count, because a panel whose rows reorder as numbers change cannot be read at
# a glance -- the eye has to find each label again on every repaint. It matches
# the order the scan summary uses on the CLI for the same reason.
STATUS_ORDER: tuple[QueryStatus, ...] = (
    QueryStatus.CLAIMED,
    QueryStatus.AVAILABLE,
    QueryStatus.UNKNOWN,
    QueryStatus.WAF,
    QueryStatus.ILLEGAL,
)


def status_style(status: QueryStatus) -> StatusStyle:
    """Look up a status, falling back rather than raising.

    A status this table has not been taught about is a display problem, and a
    display problem must never take down a running scan. It renders as a neutral
    unknown and the scan continues.
    """
    return STATUS_STYLES.get(status, STATUS_STYLES[QueryStatus.UNKNOWN])


def status_cell(status: QueryStatus) -> Text:
    """A status as it appears in the live feed: glyph, then word.

    Built as `Text` rather than a markup string because site data reaches these
    tables, and Rich parses markup in cells -- a site whose name contains
    brackets would otherwise be eaten as a style tag. The same trap `show`
    documents.
    """
    style = status_style(status)
    return Text.assemble(
        (f"{style.glyph} ", style.style),
        (style.label, style.style),
    )


# How far apart the entries of a key sit. Three spaces, not the usual two: the
# entries are a symbol and a word, and at two the symbol of the next entry
# crowds the word of the last one into "inconclusive ▲ blocked" -- one run of
# text rather than two things being named.
KEY_GAP = "   "


def status_from_name(name: str | None) -> QueryStatus:
    """A stored status string back into the enum, falling back rather than raising.

    The database holds `str(QueryStatus)`, and a row written by an older version
    can hold a spelling this one no longer knows. Reading one is a display
    problem, and a display problem must not take down the pane reporting it --
    the same rule `status_style` follows, for the same reason.
    """
    try:
        return QueryStatus(name)
    except ValueError:
        return QueryStatus.UNKNOWN


def status_key(statuses: Iterable[QueryStatus], *, columns: int = 1) -> Text:
    """The symbol-to-word key for a set of statuses: `● found   ▲ blocked`.

    Colour is never the only signal, which is why every status carries a glyph
    -- but a glyph nobody has been taught is not a signal either. The live feed
    teaches its own, because `status_cell` prints the word beside every symbol
    it draws. A two-cell column in a table cannot, and that is the whole gap
    this fills: it is the sentence the column has no room for, placed where the
    column is.

    Built from `STATUS_STYLES` rather than written out beside the table it
    explains. A key that can drift from the cells above it is worse than no key
    -- it is a wrong answer given confidently -- so the glyph, the colour and
    the word all come from the same row of the same table the cells are drawn
    from, and a status renamed there is renamed here.

    Only the glyph keeps its colour; the word is dim. The glyph is the half
    being matched by eye against the column, and a key at the same weight as
    the findings competes with them for attention it does not deserve.

    Duplicates collapse and the order is always `STATUS_ORDER`, so the key reads
    the same way every time whatever order the rows happened to arrive in.

    `columns` lays the pairs out as a grid instead of one line. Four statuses in
    a row is 44 cells, which does not fit beside anything; two by two is 27 and
    fits in the empty half of a header.
    """
    wanted = set(statuses)
    entries = [status_style(s) for s in STATUS_ORDER if s in wanted]
    if not entries:
        return Text()

    # Row-major, so the order above still reads left-to-right then down. Column
    # widths are measured across the whole grid rather than per row, which is
    # the same argument `stat_row` makes for the counters: a ragged inner edge
    # turns four pairs into eight things to find.
    rows = [entries[i : i + columns] for i in range(0, len(entries), columns)]
    widths = [
        max(len(row[col].glyph) + 1 + len(row[col].label)
            for row in rows if len(row) > col)
        for col in range(min(columns, len(entries)))
    ]

    line = Text()
    for index, row in enumerate(rows):
        if index:
            line.append("\n")
        for col, style in enumerate(row):
            if col:
                line.append(KEY_GAP)
            label = style.label
            # Every column but the last is padded out; padding the last one
            # would leave trailing blanks inside the border drawn around this.
            if col < len(row) - 1:
                label = label.ljust(widths[col] - len(style.glyph) - 1)
            line.append(f"{style.glyph} ", style=style.style)
            line.append(label, style="dim")
    return line


def stat_row(
    label: str,
    value: int,
    style: str,
    *,
    muted: bool = False,
    shown: bool | None = None,
    indent: int = 0,
) -> Text:
    """One counter line: label left, number right, both in fixed columns.

    The number is right-aligned inside its column so the digits stack -- 7, 40
    and 400 end at the same cell, which is what makes the panel comparable
    without reading it. Left-aligned, every value change shifts the eye.

    `shown` makes the row a checkbox: ☑ when the feed is drawing this kind of
    result, ☐ when it is counted but filtered out. It was strikethrough, and a
    struck number reads as crossed off -- wrong, or zero, or void -- which on
    first launch was four of the five counters before anything had run. A box
    says "filter state" in the shape every checkbox has, and leaves the number
    exactly as legible either way. None draws no box, for rows that are not
    filters.

    `indent` marks a row as a BREAKDOWN of the row above it rather than a peer
    of it -- ANALYSIS uses it for the three outcomes that sum to `extracted`.
    It is taken out of the LABEL column, never added to the line, so an
    indented row is exactly as wide as a top-level one and the numbers go on
    stacking. Indenting by prefixing would push every sub-row's value two cells
    right and break the one rule this file is least willing to break.
    """
    # Never at the cost of the whole label. A pathological indent would
    # otherwise make the width negative and `format` would stop padding.
    indent = max(0, min(indent, STAT_LABEL_WIDTH - 1))
    line = Text()
    base = "dim" if muted else ""
    number = "dim" if muted else style
    room = STAT_LABEL_WIDTH - indent
    if indent:
        line.append(" " * indent)
    if shown is not None:
        # The box takes its two cells out of the label column, like an indent.
        line.append("☑ " if shown else "☐ ", style=style if shown else "dim")
        room -= 2
    line.append(f"{label:<{room}}", style=base)
    line.append(f"{value:>{STAT_VALUE_WIDTH}}", style=number)
    return line


def value_row(
    label: str,
    value: str,
    *,
    style: str = "",
    muted: bool = False,
) -> Text:
    """A counter line whose value is a word rather than a number.

    Same total width as `stat_row`, so the model block's right edge lands in the
    same cell as the counters stacked above it -- the two read as one ruler
    rather than as two panels that happen to share a column.

    The value is measured FIRST and the label takes what is left, which is the
    opposite of `stat_row`'s fixed split. It has to be: `38 tok/s` is wider than
    the six cells a count needs, and a value that overflows its column would
    push the right edge out on exactly the rows that are changing fastest. The
    label is the half that can be truncated without losing the number.
    """
    room = max(0, STAT_LABEL_WIDTH + STAT_VALUE_WIDTH - len(value) - 1)
    line = Text()
    line.append(f"{_elide(label, room):<{room}} ", style="dim" if muted else "")
    line.append(value, style="dim" if muted else style)
    return line


def model_name(key: str, width: int) -> str:
    """A model key as a name someone recognises, in the width available.

    Model keys are paths -- `unsloth/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf`
    is one string, and llama-server's router mode reports them in full. The part
    that identifies the model is at the END, so plain truncation is exactly
    backwards: it keeps the vendor and the repo, which are identical for every
    model pulled from one place, and cuts the size and the quantisation, which
    are what someone is choosing between. The directory prefix and the `.gguf`
    suffix go first, and only what is left is elided.
    """
    name = key.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    if name.lower().endswith(".gguf"):
        name = name[: -len(".gguf")]
    return _elide(name or key, width)


def throughput_label(tokens_per_second: float | None) -> str:
    """Generation speed, at the precision the number is actually good to.

    Whole tokens per second: this is an average over finished requests, and a
    decimal place on an average of three would imply a measurement it is not.
    """
    if not tokens_per_second:
        return ""
    return f"{tokens_per_second:.0f} tok/s"


# Spinner frames, advanced one per redraw. At REDRAW_INTERVAL that is a full
# turn every second -- fast enough to read as "working", slow enough not to
# strobe. Braille rather than the ASCII |/-\ because every frame occupies one
# cell of the same weight, so the line beside it does not appear to twitch.
SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

# Column widths for the startup lines. Narrower than the counter columns: this
# block sits in the same 26-cell column but carries three fields, not two.
PHASE_LABEL_WIDTH = 9
PHASE_STATE_WIDTH = 8


def spinner(tick: int) -> str:
    return SPINNER_FRAMES[tick % len(SPINNER_FRAMES)]


def phase_line(
    label: str,
    state: str,
    *,
    elapsed: float = 0.0,
    tick: int = 0,
) -> Text:
    """One startup step, as a line that replaces itself rather than repeating.

    A model that takes three minutes to load -- and a cold first load was
    measured at 187s -- has to say so while it happens. Printed as log lines it
    would either be one silent gap or a wall of "still loading"; as a single
    line that mutates in place, it animates while it works and then states the
    outcome and how long it took, in the same cell.

    The elapsed time keeps running while `working` because that is the number
    someone is actually asking for at that moment: not "is it stuck" but "how
    much longer". It freezes at the total once the step lands.

    USED FOR PASS 2 AS WELL as the two startup steps, which is why `building`
    and `cached` are in the table. Synthesis is the same shape of event -- one
    slow step, a spinner while it runs, an outcome and a duration when it lands
    -- and the results pane already reuses this function for the rebuild path.
    A second vocabulary for the third instance of one event is how two screens
    start describing the same thing differently.

    `cached` is a landed state, not a failure: the profile is there, the model
    was never asked. It keeps the green dot and dims the word, because what
    happened is worth reading once and is not worth the weight of `ready`.
    """
    # Keyed on the word that is actually printed, so the state a caller passes
    # and the state a reader sees cannot drift apart.
    glyph, glyph_style, word_style = {
        "loading": (spinner(tick), "bold cyan", "none"),
        # Same animation as loading, because it is the same answer to the same
        # question -- "is this still moving" -- for a step that is fetching
        # rather than starting. It is a separate word only because the two take
        # very different amounts of time and saying which one you are waiting
        # on is the difference between patience and a killed process.
        "installing": (spinner(tick), "bold cyan", "none"),
        "building": (spinner(tick), "bold cyan", "none"),
        "ready": ("●", "bold green", "green"),
        "cached": ("●", "green", "dim"),
        "failed": ("✕", "bold red", "red"),
        "waiting": ("·", "dim", "dim"),
    }.get(state, ("·", "dim", "dim"))

    line = Text()
    line.append(f"{label:<{PHASE_LABEL_WIDTH}}", style="dim")
    line.append(f"{glyph} ", style=glyph_style)
    line.append(f"{state:<{PHASE_STATE_WIDTH}}", style=word_style)
    # Sub-second steps read as "<1s", not "0s". A browser that came up in six
    # tenths of a second reporting "0s" looks like a step that never ran, and
    # the exact figure below a second is not a number anyone acts on.
    if not elapsed:
        stamp = ""
    elif elapsed < 1:
        stamp = "<1s"
    else:
        stamp = elapsed_label(elapsed)
    line.append(f"{stamp:>5}", style="dim")
    return line


# Width of the field column in an anchor line. Fixed so values stack.
ANCHOR_FIELD_WIDTH = 10


def anchor_line(anchor: Any, *, width: int) -> Text:
    """One anchor as it appears in the scan pane's narrow left column.

    Field and value on one line, the value dimmed: the field is what you scan
    the list for, the value is what you check once you have found it. Truncated
    with an ellipsis rather than wrapped -- a wrapped anchor would make the
    block's height depend on the length of what someone typed, which is exactly
    the instability capping the list is there to avoid.
    """
    line = Text()

    # No trust glyph. Every anchor this screen can show was made by the editor,
    # which no longer asks for a trust level, so the column was the same
    # character on every row -- a signal carrying no information, which is the
    # thing the rest of this file is careful not to draw.
    #
    # A FIXED field column, so every value starts in the same cell. Sizing it to
    # each field's own length was the obvious thing and it is wrong: the values
    # then began at a different column on every row, which is the ragged edge
    # this file's own rules exist to prevent. A field longer than the column is
    # truncated rather than allowed to push its value out of line.
    field = _elide(str(anchor.field), ANCHOR_FIELD_WIDTH)
    line.append(f"{field:<{ANCHOR_FIELD_WIDTH}} ")
    line.append(
        # The two cells the glyph used to occupy are the value's now.
        _elide(str(anchor.value), max(0, width - ANCHOR_FIELD_WIDTH - 1)),
        style="dim",
    )
    return line


def _elide(text: str, room: int) -> str:
    """Trim to `room` cells, marking that something was cut."""
    if room <= 0:
        return ""
    if len(text) <= room:
        return text
    return text[: max(0, room - 1)] + "…"


def count_of(quantity: int, singular: str, plural: str | None = None) -> str:
    """"1 site", "2 sites" -- the number and its noun agreeing.

    Trivial, and worth having in one place: "1 sites checked" is the kind of
    detail that makes a careful tool look careless, and it appears wherever a
    count is printed beside a word, which is most of this interface.
    """
    if quantity == 1:
        return f"{quantity} {singular}"
    return f"{quantity} {plural or singular + 's'}"


def progress_bar(completed: int, total: int, width: int) -> Text:
    """A determinate bar in block characters, exactly `width` cells wide.

    Drawn rather than delegated to a widget, for the reason the stylesheet
    records: a compound progress widget sizes its own parts and collapsed to
    nothing inside a one-row strip. This returns a fixed number of cells, so the
    bar's ends line up with the panes above it on every repaint.

    A heavy rule rather than a filled block: it reads as an instrument rather
    than a download, and the unfilled half stays legible instead of becoming a
    grey slab. Zero total renders as all-empty rather than dividing by it -- an
    idle scan looks idle.

    The two halves are different CHARACTERS, not one character in two colours.
    Weight alone survives a monochrome terminal, a screenshot and a colour-blind
    reader; colour alone would render every bar identical to a finished one --
    the same rule the status glyphs follow, and a progress bar that always looks
    complete is a worse lie than most.
    """
    width = max(0, width)
    filled = 0 if total <= 0 else min(width, round(width * completed / total))
    bar = Text()
    bar.append("━" * filled, style="bold cyan")
    bar.append("─" * (width - filled), style="dim")
    return bar


def elapsed_label(seconds: float) -> str:
    """A duration as an operator reads it, not as a float.

    Seconds below a minute, m:ss above it. `132.4s` forces mental arithmetic
    during the exact moments someone is deciding whether a scan has hung.
    """
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, remainder = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{remainder:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def response_label(seconds: float | None) -> str:
    """How long ONE site took, at the scale one site actually takes.

    Separate from `elapsed_label`, which measures a whole scan and rounds to
    whole seconds -- most site responses are under a second, so through that
    function nearly every row in the findings table would read `0s`. This is a
    different quantity at a different magnitude, and sharing one formatter
    between them would flatten the faster half of the results to zero.

    Empty rather than `0ms` when there is no timing at all. A row restored from
    the database has a verdict and no duration, and printing a number there
    would claim the request was instant when the truth is that no request was
    made on this run.
    """
    if seconds is None:
        return ""
    if seconds < 1:
        return f"{round(seconds * 1000)}ms"
    if seconds < 10:
        return f"{seconds:.1f}s"
    return f"{seconds:.0f}s"


# The stylesheet. Kept here rather than on each screen so spacing is decided
# once: the page gutter is 2 cells and the gutter between panes is 1, and every
# screen inherits both instead of re-picking them.
#
# Sizing note carried over from the settings screen, which paid for it: panels
# that must sit under one another get `height: auto`, and exactly one element per
# screen gets `1fr`. Give two siblings `1fr` and the slack is split between them,
# which reads as a hole in the middle of the layout rather than as breathing room
# at the bottom.
SHERLOCK_CSS = """
Screen { background: $surface; }

/* ---- chrome ---------------------------------------------------------- */

/* The identity strip. Docked so it survives every screen switch: an operator
   console that loses its "which database am I writing to" line when a tab
   changes is how the wrong database gets written to. */
#appbar {
    dock: top;
    height: 1;
    padding: 0 2;
    background: $panel;
    color: $text-muted;
}
#appbar-title { text-style: bold; color: $accent; }

TabbedContent { height: 1fr; }
TabPane { padding: 0 2; }

/* ---- shared furniture ------------------------------------------------ */

.pane-title {
    color: $accent;
    text-style: bold;
    padding: 1 0 0 0;
    height: 2;
}
.dim { color: $text-muted; }
.hint { color: $text-muted; padding: 0 0 1 0; }

/* A framed block. `round` rather than `heavy`: at one cell wide a heavy border
   competes with the content for attention, and this screen has a lot of blocks. */
.panel {
    border: round $panel-lighten-2;
    padding: 0 1;
    height: auto;
}
.panel-accent { border: round $accent; }

/* Buttons wear the app's colour, not Textual's. `variant="primary"` resolves to
   $primary -- #0178D4, a blue that appears nowhere else in this chrome, where
   every title, active tab and affordance is $accent. Left at the default, SCAN
   and `Build profile` were the only blue objects on screen, which is how a
   deliberate palette ends up looking like an unfinished one.
   The token itself is left alone: $primary still does honest work as the
   selection wash on `.row-selected` and `.counter-row:focus`, where a tint is
   wanted and an identity is not. Only the buttons are restated.
   `color` is deliberately not set -- Textual's `auto 87%` picks the contrasting
   foreground for whatever background lands here, which measures 153/255 against
   this amber. Focus still reverses, because nothing here sets `text-style`. */
Button.-primary {
    background: $accent;
    border-top: tall $accent-lighten-3;
    border-bottom: tall $accent-darken-3;
}
Button.-primary:hover {
    background: $accent-darken-2;
    border-top: tall $accent;
}
Button.-primary.-active {
    background: $accent;
    border-top: tall $accent-darken-3;
    border-bottom: tall $accent-lighten-3;
}

/* THE BUTTON GRAMMAR. Three kinds, and what a control looks like says which:

   - Primary (`variant="primary"`, raised amber, above): starts work. One per
     view -- SCAN, Build profile, a dialog's Done.
   - Chip (`.chip`, flat one-row panel): changes state, opens something, or
     points somewhere. Toggles, Actions, Add anchor, Rebuild.
   - Destructive (`variant="error"`, raised red): only inside a confirmation.

   Before this, raised grey buttons sat beside amber ones and read as disabled,
   and a flat red "Delete" sat in a header as the loudest thing on screen. */
.chip {
    height: 1;
    min-width: 0;
    width: auto;
    border: none;
    padding: 0 1;
    background: $panel;
    color: $text;
    text-style: none;
}
.chip:hover { background: $panel-lighten-2; }
/* Dialogs are painted $panel, so a chip there needs one step up to show. */
#dialog .chip { background: $panel-lighten-1; }
#dialog .chip:hover { background: $panel-lighten-2; }
.chip:disabled { color: $text-disabled; }

/* ONE FOCUS STYLE, app-wide: an amber tint and bold label. Textual's default
   reverses the label, and on a restyled button that drew a pale chip INSIDE
   the button -- it looked like a rendering fault (Cancel in the delete dialog,
   "Check N new" in the resume dialog). The tint reads on flat chips and raised
   buttons alike, and is the wash `.counter-row:focus` already used. */
Button:focus {
    text-style: bold;
    border-top: tall $accent;
    border-bottom: tall $accent;
}
Button.-primary:focus { background: $accent-lighten-1; border-top: tall $accent-lighten-3; border-bottom: tall $accent-darken-3; }
/* `Button.` on the front is load-bearing. A pseudo-class weighs the same as a
   class, so `Button:focus` above (type + pseudo) outranked a bare `.chip:focus`
   and gave these ONE-ROW controls a tall top and bottom border on focus. The
   two borders took the only row there is: clicking `verbose` or `found only`
   left an empty amber bar with no label until focus moved elsewhere. Chips are
   flat; their focus is a tint and a bold label, never a border. */
Button.chip:focus, Button.toggle:focus {
    border: none;
    text-style: bold;
    background-tint: $accent 25%;
}

/* Scrollbars: one cell, panel-coloured. The defaults are two cells of solid
   colour, which drew a black slab beside a log with one line in it. */
* {
    scrollbar-size-vertical: 1;
    scrollbar-size-horizontal: 1;
    scrollbar-background: $surface;
    scrollbar-background-hover: $surface;
    scrollbar-background-active: $surface;
    scrollbar-color: $panel-lighten-2;
    scrollbar-color-hover: $panel-lighten-3;
    scrollbar-color-active: $accent;
    scrollbar-corner-color: $surface;
}
/* The footer scrolls sideways when its keys outrun the width, and hides that
   scrollbar on purpose -- the rule above would otherwise draw one over it. */
Footer { scrollbar-size: 0 0; }

/* Hover text, app-wide. Textual ships a `Tooltip` already; these three rules
   are the ones its defaults get wrong on a screen this dense.

   A BORDER, because the default box is `background: $panel` -- the same paint
   as the toggles and the appbar it floats over, so it read as part of the
   layout rather than as something that had just appeared. Accent rather than
   the muted border `.panel` uses: this is the one thing on screen the pointer
   is asking about.

   `max-width: 52` rather than the default 40. These carry two sentences, and at
   40 cells minus padding the second one broke into a narrow column five lines
   tall. `padding: 0 1` rather than `1 2` for the same reason -- with a border
   the default padding spends four rows and four columns on air. */
Tooltip {
    max-width: 52;
    padding: 0 1;
    border: round $accent;
}

/* ---- scan pane ------------------------------------------------------- */

/* The target row. A grid, not a horizontal box, because the label, the field
   and the button must hold their columns whatever the terminal width -- a
   button that slides left as the window narrows is the thing that gets
   mis-clicked.

   `margin` for the gap above, never `padding`. Padding is drawn INSIDE the
   fixed height, so `height: 3; padding: 1` leaves two rows for a three-row
   button and clips its label off the bottom -- the button rendered as an empty
   block with no word on it, which is exactly as useless as it sounds. */
#target-row {
    layout: grid;
    grid-size: 3 1;
    grid-columns: 10 1fr 16;
    grid-gutter: 0 1;
    height: 3;
    margin: 1 0 0 0;
}
/* Vertically centred against the bordered field beside it. Aligned to the top
   it sits level with the field's border rather than with the text in it. */
#target-label {
    height: 3;
    content-align: left middle;
    color: $text-muted;
    text-style: bold;
}
#target-input { width: 100%; }

/* The per-run options row. Same 10-cell label column as the target row above
   it, so OPTIONS and TARGET share a left edge and the controls under each line
   up rather than stepping in and out. */
#options-row {
    layout: grid;
    grid-size: 4 1;
    grid-columns: 10 14 13 1fr;
    grid-gutter: 0 1;
    height: 1;
    margin: 1 0 0 0;
}
#options-label { height: 1; color: $text-muted; text-style: bold; }
/* Flat toggles, not raised buttons. They report state as much as they invite a
   press, and a full button chrome around each would out-shout SCAN, which is
   the control that actually starts something. */
.toggle {
    height: 1;
    min-width: 0;
    border: none;
    background: $panel;
    color: $text;
    padding: 0 1;
}
.toggle:hover { background: $panel-lighten-2; }
#scan-config { height: 1; color: $text-muted; }

/* Counters left, live feed right. The counters column is fixed rather than
   proportional: it holds a known amount of text, and letting it grow with the
   window would stretch the gap between each label and its number until they
   stop reading as pairs. Every extra cell goes to the feed, which can use it. */
#scan-body {
    layout: grid;
    grid-size: 2 1;
    grid-columns: 26 1fr;
    grid-gutter: 0 2;
    height: 1fr;
    /* Air between the controls and the readouts. Without it OPTIONS and SITES
       sat on consecutive lines and read as one block, so the toggles looked
       like another counter rather than something you press. */
    margin: 1 0 0 0;
}
/* A rule, not a gap, between the two columns. At this density whitespace alone
   stopped reading as a division -- the counter labels and the feed's first
   column looked like one ragged table. */
#counters-col {
    height: 1fr;
    border-right: solid $panel-lighten-2;
    padding: 0 1 0 0;
}
#feed-col { height: 1fr; }
/* Compact (see COMPACT_WIDTH): the counters column gives back three cells --
   21 for the rows plus the rule and its padding -- and the options row its
   label column. */
ScanPane.-compact #scan-body { grid-columns: 23 1fr; }
/* Sized to content and empty before a scan, so an idle pane shows no startup
   block at all rather than reserving a hole for one. */
#startup { height: auto; }
#counters { height: auto; }

/* Each counter is a control. The hover tint is the only thing that says so
   before it is clicked, and the focus outline is what says so to someone
   arriving by Tab -- without either, five clickable lines look like five
   labels. */
.counter-row { height: 1; }
.counter-row:hover { background: $panel-lighten-2; }
.counter-row:focus { background: $primary 30%; }

/* The anchors section. Height follows its content, which is bounded by the
   preview cap, so turning analysis on moves the counters down once by a known
   amount rather than by however much setup someone has done. */
#anchors-block { height: auto; padding: 0 0 1 0; }
/* Title left, + hard right, so the button sits on the column's edge instead of
   floating wherever the word happens to end. */
#anchors-head {
    layout: grid;
    grid-size: 2 1;
    grid-columns: 1fr 3;
    height: 1;
}
#anchors-title { color: $text-muted; text-style: bold; height: 1; }
/* A one-cell affordance, not a chrome button: it sits inside a readout column
   and a raised border around a single "+" would out-shout the counters. */
#add-anchor {
    height: 1;
    min-width: 3;
    width: 3;
    border: none;
    padding: 0;
    background: $panel;
    color: $accent;
    text-style: bold;
}
#add-anchor:hover { background: $panel-lighten-2; }
#anchors-list { height: auto; padding: 0 0 0 0; }
#counters-title, #feed-title {
    color: $text-muted;
    text-style: bold;
    height: 1;
}
/* Findings get twice the height of the activity log beneath them. They are the
   evidence and they are what someone is watching for; the log is context and
   only its last few lines matter at any moment. An even split made the feed --
   the reason the screen exists -- look like half a footnote. */
/* Clipped rather than scrolled sideways: the link is the last column and Enter
   opens it in full, while a horizontal scrollbar cost a row of a short pane. */
#feed { height: 2fr; overflow-x: hidden; }
/* A rule above ACTIVITY rather than a box around each panel.
   Boxes were considered and rejected: four framed panels on one screen reads
   as a form, not an instrument, and every border costs a cell of width and a
   row of height -- the feed loses URL room and the 26-cell counters column can
   least afford it. The screen already separates things with a heading and a
   rule (the vertical one between the columns), so this uses the same language
   for the one boundary that was genuinely unclear: FINDINGS and ACTIVITY sit
   in the same column with nothing between them. One row, no width cost. */
#activity-title {
    border-top: solid $panel-lighten-2;
    color: $text-muted;
    text-style: bold;
    height: 2;
    padding: 0;
}
#activity { height: 1fr; background: $surface; color: $text-muted; }

/* The AI block sits under the counters, in the same column, separated by a
   rule. Same column because both answer "how is this run going"; separated
   because one counts sites and the other counts model jobs, and stacking them
   flush made the numbers look like one list. */
#ai-block { height: auto; padding: 1 0 0 0; }

/* Pass 2, set off from the Pass 1 tallies above it. `margin`, never `padding`
   -- padding is drawn INSIDE the height, which on a one-row line leaves zero
   rows for the text and the line simply vanishes. That trap has been paid for
   twice on this screen already (the options hint, the target button).
   `height: auto` rather than 1 so hiding the row takes its margin with it,
   instead of leaving a gap where Pass 2 has not been reached. */
#synthesis-line { height: auto; margin: 1 0 0 0; }

/* What is doing the analysing, directly beneath what the analysis has produced.
   Below rather than above the ANALYSIS counts on purpose: the counts are read
   continuously during a run, the model's name and window are read once when
   deciding whether to trust them, and the block nearer the top of a column is
   the one the eye returns to. Hidden entirely when analysis is off, like the
   ANCHORS block -- a readout describing work this run will not do is noise. */
#model-block { height: auto; padding: 1 0 0 0; }
#model-lines { height: auto; }

/* The progress strip is drawn as text rather than with a `ProgressBar` widget.
   The widget is a compound of bar, percentage and ETA whose parts size
   themselves, and inside a one-row strip it collapsed to zero width -- the
   counts appeared with no bar at all. A bar built from block characters is one
   `Static`, sizes exactly as told, and lines its ends up with the panes above
   it, which a widget that measures itself will not reliably do. */
#progress-strip { height: 1; }

/* ---- results pane ---------------------------------------------------- */

/* Username list left, detail right -- the master/detail shape, because the
   question is always "this one, tell me more".

   The list column is a FIXED width, not a fraction. Its columns add up to a
   known number of cells, and at `1fr` of a narrow terminal the total came out
   under that: the last column was clipped to "si" and the count it held could
   not be read at all. Fixed, the list always fits and the detail absorbs the
   slack -- which is the right way round, since prose reflows and a table of
   numbers does not.

   35 is that total, and it is arithmetic rather than taste: 31 + 4. The cells
   come to 31 (17 username + 7 found + 7 sites -- each column's width plus one
   cell of padding on each side), and the 4 is what the widget spends around
   them: 1 on its right padding, 1 on its border, and 2 on the vertical
   scrollbar a list longer than the pane puts there. The original 34 did not
   count that scrollbar, so a database with 40 usernames in it left the columns
   30 cells and grew a HORIZONTAL scrollbar under a table of three short
   columns.

   Every cell past that goes to the detail, and the strip in it has a floor: at
   40 an 80-column terminal leaves the section tabs 32 cells for 33 of them, and
   `Tabs` is a scrolling strip -- squeezed, it drops tabs silently and from the
   LEFT rather than wrapping or ellipsizing. Measured when a delete column lived
   here; the column is gone and the floor is not. */
#results-body {
    layout: grid;
    grid-size: 2 1;
    grid-columns: 37 1fr;
    grid-gutter: 0 2;
    height: 1fr;
}
/* Narrow: the list becomes a picker over the detail (ctrl+l), so the record
   gets the full width. See NARROW_WIDTH in results_pane.py. */
ResultsPane.-narrow #results-body { grid-size: 1 1; grid-columns: 1fr; }
ResultsPane.-narrow .pane-title { display: none; }
ResultsPane.-narrow #username-list { border-right: none; }
ResultsPane.-narrow #username-list { display: none; }
ResultsPane.-narrow.-picking #username-list { display: block; }
ResultsPane.-narrow.-picking #result-detail { display: none; }
#username-list {
    height: 1fr;
    border-right: solid $panel-lighten-2;
    padding: 0 1 0 0;
}
#result-detail { height: 1fr; }
/* Identity left, actions right: two lines of who-and-what, and a one-row
   chip pinned to the right edge of the first. */
#detail-head {
    layout: grid;
    grid-size: 2 1;
    grid-columns: 1fr auto;
    grid-gutter: 0 2;
    height: auto;
    padding: 0 0 1 0;
}
#detail-header { height: auto; }
#record-actions { margin: 0; }

/* The section strip. Styled DOWN from the tab bar above it, not hidden: two
   bars with the same weight read as two competing navigations, which is what
   made this pane look improvised.
   `#tabs-list-bar` is the container holding the tabs, NOT the underline
   decoration -- hiding it takes the whole strip with it and leaves a blank
   line where the sections should be. The indicator is toned down instead: the
   active segment keeps the accent, the rest of the bar is painted the same as
   the pane so it reads as a marker under one word rather than a rule across
   the pane. */
#detail-tabs {
    height: 2;
    background: $surface;
}
#detail-tabs Tab {
    padding: 0 2 0 0;
    color: $text-muted;
    text-style: none;
}
#detail-tabs Tab.-active { color: $accent; text-style: bold; }

/* The underline is what makes these read as TABS rather than as a row of
   words. It was hidden entirely at first, which left the active section marked
   only by colour -- and then only while the strip had focus, where Textual's
   block cursor painted it as a solid highlighted block that looked like a
   selected list row, not a tab.
   `.underline--bar` carries both halves: `color` is the marker under the
   active tab, `background` is the track running the full width. The track is
   painted as the pane so only the marker shows, which is the difference
   between a tab underline and the full-width rule under the main nav.
   `Underline` is the decoration; `#tabs-list-bar` is the container that HOLDS
   the tabs -- hiding that one by mistake takes the whole strip with it. */
#detail-tabs .underline--bar { color: $accent; background: $surface; }

/* Focus must not restyle the active tab. Textual's default turns it into a
   block-cursor row -- reversed colours across the whole label -- so the strip
   changed appearance depending on where focus happened to be. */
#detail-tabs:focus Tab.-active {
    color: $accent;
    background: $surface;
    text-style: bold;
}
#detail-tabs:focus .underline--bar { color: $accent; background: $surface; }

/* The filter, what it is hiding, and the key -- one line above the table on
   wide panes. `auto` columns size to the chip and the key; the counts take the
   rest and WRAP rather than clip, because the unresolved count is the one
   number on this line that must survive a narrow terminal. */
#sites-controls {
    layout: grid;
    grid-size: 3 1;
    grid-columns: auto 1fr auto;
    grid-gutter: 0 2;
    height: auto;
}
#sites-counts { height: auto; }
#sites-key { height: 1; width: auto; }
/* Too narrow for one line: the key takes a row of its own under the counts. */
#sites-controls.-stacked { grid-size: 2; grid-columns: auto 1fr; }
#sites-controls.-stacked #sites-key { column-span: 2; }

/* Exactly one scroll region visible at a time -- the whole point of switching
   rather than stacking. `overflow-x: hidden` because a long URL would otherwise
   give the table a horizontal scrollbar on top of its vertical one; the cells
   ellipsize instead, and Enter opens the full link anyway. */
#detail-switch { height: 1fr; }
#sec-sites {
    height: 1fr;
    overflow-x: hidden;
}
#sec-profile { height: 1fr; }
#detail-profile { height: auto; }

/* Only mounted visible when there is no profile to show, so the section is a
   viewer the rest of the time. */
#profile-actions { height: auto; padding: 0; }
#profile-anchor-line { height: auto; padding: 0; }
#profile-status { height: auto; padding: 0 0 1 0; }
#profile-buttons {
    height: auto;
    width: 100%;
    margin: 1 0 0 0;
}
/* Left-aligned, directly under the card it acts on, and sized to its label. */
#profile-build { width: auto; margin: 0 2 0 0; }
/* The chip sits on the middle row of the raised button beside it. */
#profile-anchors { margin: 1 0 0 0; }
#profile-actions.-has-profile #profile-anchors { margin: 0 2 0 0; }
/* A blank line between what the profile is and what can be done to it. Flush,
   the chips read as a third line of the status sentence. */
#profile-actions.-has-profile #profile-buttons { margin: 1 0 0 0; }
#profile-actions.-has-profile #profile-build { margin: 0 2 0 0; }
#profile-analyse { margin: 1 0 0 0; }
#profile-actions.-has-profile #profile-analyse { margin: 0; }
#profile-actions.-has-profile { padding: 0 0 1 0; border-bottom: solid $panel-lighten-2; margin: 0 0 1 0; }

/* The POINTER state: no stored evidence, so the only thing on offer is a trip
   to the scan tab. It is not a commit and it must not look like one -- the
   chip, in amber, rather than the primary block SCAN wears. */
#profile-actions.-no-evidence #profile-build {
    height: 1;
    min-width: 0;
    border: none;
    padding: 0 1;
    background: $panel;
    color: $accent;
    text-style: bold;
}
#profile-actions.-no-evidence #profile-build:hover { background: $panel-lighten-2; }
/* The ID selector above outranks the app-wide focus rule, so the pointer
   needs its own -- without it the only cue was an 8/255 background shift. */
#profile-actions.-no-evidence #profile-build:focus {
    background: $accent 30%;
    color: $text;
    text-style: bold;
}

/* ---- settings pane --------------------------------------------------- */

#settings-rows { height: auto; max-height: 1fr; }
.section { color: $accent; text-style: bold; padding: 1 0 0 1; height: 2; }
.row { padding: 0 1; height: 1; }
.row-selected { background: $primary 30%; }
.help { padding: 1 1 0 1; height: auto; min-height: 3; }
.paths { padding: 1 1 0 1; color: $text-muted; }

/* ---- dialogs --------------------------------------------------------- */

ModalScreen { align: center middle; }
#dialog {
    background: $panel;
    border: round $accent;
    padding: 1 2;
    width: 90%;
    max-width: 96;
    height: auto;
}
.dialog-title { text-style: bold; color: $accent; }
#models { height: auto; max-height: 20; margin: 1 0; }

/* ---- extractions panel ----------------------------------------------- */

/* Master-detail inside one section: the site list left, the extraction right.
   The list column is FIXED and the detail absorbs the slack -- the same rule
   `#results-body` states one level up, and for the same reason. Its two columns
   add up to a known 21 cells; at a fraction of a narrow terminal the total came
   out under that and the site names clipped, while the reading beside it is
   prose that reflows happily into whatever is left.

   28 IS MEASURED, NOT PICKED. A `DataTable` pads every cell by one on each
   side, so its two columns occupy (5+2) + (15+2) = 24 cells, and this container
   spends 1 on its own right padding plus 1 on the border rule -- so anything
   under 26 gives the table a HORIZONTAL scrollbar, which is what a too-narrow
   column actually produces rather than the ellipsis one might expect. 28 leaves
   two cells of headroom. Re-measure if either column width changes. */
#sec-extractions {
    layout: grid;
    grid-size: 2 1;
    grid-columns: 28 1fr;
    grid-gutter: 0 2;
    height: 1fr;
}
/* A rule rather than a gap, matching the two other master-detail splits in this
   app. At this density whitespace alone stops reading as a division. */
#extraction-list-col {
    height: 1fr;
    border-right: solid $panel-lighten-2;
    padding: 0 1 0 0;
}
/* The list takes what is left after the summary, rather than the other way
   round: the summary is three lines of known height and the list is the part
   that should grow with the terminal. */
#extraction-list { height: 1fr; }
/* Set off from the list by a rule, because it is a statement ABOUT the list
   rather than another row of it -- flush against the last site it read as one. */
#extraction-summary {
    height: auto;
    border-top: solid $panel-lighten-2;
    padding: 1 0 0 0;
}
/* The one scroller in this section besides the list, and deliberately so: the
   reasoning is prose of unbounded length and has to be reachable. `overflow-x`
   stays hidden -- these are sentences, and a horizontal bar under a paragraph is
   unreadable. They wrap instead. */
#extraction-detail-col {
    height: 1fr;
    overflow-x: hidden;
}
#extraction-detail { height: auto; }

/* ---- anchor editor --------------------------------------------------- */

#anchor-blurb { height: auto; padding: 0 0 1 0; }
#anchor-list { height: auto; max-height: 10; margin: 0 0 1 0; }
/* field | value | Add, on one row. Every control in it is three cells tall, so
   Add lines up with the boxes it submits rather than floating beside them. */
#anchor-form { height: auto; }
#anchor-field { width: 24; margin: 0 1 0 0; }
#anchor-value { width: 1fr; margin: 0 1 0 0; }
#anchor-form Input { border-title-color: $text-muted; }
#anchor-form Input:focus { border-title-color: $accent; }
#anchor-add { min-width: 8; }
#anchor-status { height: auto; padding: 0 0 0 1; }
/* The footer: the list's own action on the left, the dialog's on the right.
   Both full-size buttons -- one size in one dialog -- and only Done is amber,
   because a dialog has one way out that it recommends. */
#anchor-buttons { height: auto; margin: 1 0 0 0; }
#anchor-buttons .spacer { width: 1fr; height: 1; }

/* ---- record actions menu --------------------------------------------- */

/* A popup at the top right, under the chip that opens it, not a centred
   dialog: it is a menu, and a menu appears where it was asked for. */
RecordActionsScreen { align: right top; background: $background 40%; }
#actions-menu {
    width: auto;
    height: auto;
    margin: 6 3 0 0;
    background: $panel;
    border: round $panel-lighten-3;
    padding: 0 1;
}
#actions-title { color: $text-muted; padding: 0 1; }
#actions-list { width: auto; height: auto; border: none; background: $panel; padding: 0; }

/* ---- update dialog --------------------------------------------------- */

#update-detail { height: auto; padding: 1 0; }
/* Absent until the install starts, so the offer state is not padded out by a
   reserved hole where progress will go. */
#update-status { height: auto; padding: 1 0 0 0; }
/* Same shape as the confirm dialog's row: spacer, then the pair. The wider
   second column takes "Update"/"Close" without either clipping. */
#update-buttons {
    layout: grid;
    grid-size: 3 1;
    grid-columns: 1fr 12 14;
    grid-gutter: 0 2;
    height: 3;
}

/* ---- confirm dialog -------------------------------------------------- */

#confirm-detail { height: auto; padding: 1 0; }
/* Buttons right-aligned, cancel first. The destructive one is furthest from
   where the eye lands and is not the one focused on open. */
#confirm-buttons {
    height: auto;
    align: right middle;
}
/* A matched pair at one fixed width, whatever the labels say. Sized to the
   label, a long name in "Delete <name>" made the destructive button the
   widest thing in the dialog; the name now lives in the title, which wraps. */
#confirm-buttons Button { width: 14; min-width: 14; margin: 0 0 0 2; }
/* Danger: the frame and the title go red, so a dialog that destroys something
   does not wear the same amber as one that asks which sites to check. */
ConfirmScreen.-danger #dialog { border: round $error; }
ConfirmScreen.-danger .dialog-title { color: $text-error; }

/* ---- resume dialog --------------------------------------------------- */

#resume-detail { height: auto; padding: 1 0; }
/* Widest button first after the spacer: the cheapest, non-destructive action
   sits nearest the eye, with "re-scan everything" further out. */
#resume-buttons {
    layout: grid;
    grid-size: 4 1;
    grid-columns: 1fr 18 20 12;
    grid-gutter: 0 2;
    height: 3;
}

/* Four real buttons instead of three and a spacer, which the `1fr` column
   cannot absorb: it is sized from what the fixed columns leave over, and at 80
   cells they leave 10 -- enough to render "Analyse" and drop the count and the
   noun. Every column is fixed here so the shortfall is shared instead of
   landing entirely on the one button whose label carries a number. */
#resume-buttons.-analysing {
    grid-columns: 15 14 17 12;
}

/* DataTable: no zebra striping on the live feed. Stripes imply the rows are
   uniform records to be scanned in bulk; these rows are findings, and the
   status colour is doing the differentiating already. Two competing row
   treatments read as noise. */
DataTable { background: $surface; }
DataTable > .datatable--header {
    background: $surface;
    color: $text-muted;
    text-style: bold;
}
"""
