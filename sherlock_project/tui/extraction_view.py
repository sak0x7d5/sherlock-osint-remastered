"""How one site's Pass 1 extraction is drawn, and how a set of them is ordered.

Render functions rather than a screen. This started life as a modal dialog, one
site per open, and the dialog is gone: reviewing extractions is a SWEEP, not an
inspection. A developer checking whether a prompt edit landed reads down a list;
a user judging whether a model is worth keeping reads the distribution. Both want
to move a cursor and watch the detail change, and neither wants to open and close
149 dialogs to do it.

So the panel in the results pane owns the layout and calls these, which keeps
one description of what an extraction looks like. Two renderings of the same
record would drift, and the drift would be invisible -- each would look
reasonable on its own.

**Two readers, and the split is the two blocks.** A user asks "did the model find
anything worth having here", which `extraction_facts` answers. A developer asks
"why did it decide that", which `extraction_reasoning` answers -- and that block
is the half that moves when `resources/pass_one.md` is edited, which is the only
way to see a prompt change land on a real page.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from rich.text import Text

from sherlock_project.database import SiteExtractionRecord
from sherlock_project.tui.theme import model_name

# How wide a model key may be on the provenance line. Narrower than the 48 the
# old dialog used: this now renders into the panel's detail column, which is the
# slack half of an already-split pane rather than a 96-cell modal.
MODEL_LINE_WIDTH = 40

# This module has its own two rulers, and deliberately does NOT borrow
# `value_row` / `stat_row` from the theme. Those are built for the scan pane's
# 26-cell column, where the total is 19 cells and the VALUE is measured first so
# a widening number cannot push the right edge out. Here the values are a model
# key, a timestamp and a sentence -- all longer than that whole budget -- so
# through `value_row` the label got whatever was left, which was nothing:
# `model` rendered as `mo…`, and `scanned` and `contract` vanished entirely,
# leaving three rows with three different left edges.
#
# The fix is the ordinary one for a wide column: fix the LABEL and let the value
# take the rest.
PROVENANCE_LABEL_WIDTH = 10
# Wider than the scan pane's 13, because extraction keys are model-chosen and
# routinely longer than a counter label -- `conference_talks` is 16 cells.
FIELD_LABEL_WIDTH = 18


def current_contract_hash() -> str | None:
    """What Pass 1 hashes to right now, or None if it cannot be worked out.

    Imported lazily and guarded, for two separate reasons. `ai_engine` is a
    heavyweight import that pulls the whole AI stack in, and the results pane is
    otherwise a database viewer that must open without it; and the hash reads
    `resources/pass_one.md` off disk, which can fail in a way that is not worth
    losing the extraction panel over. None simply means the staleness line is
    omitted rather than shown wrongly -- there is no honest fallback for "is this
    extraction current", so it says nothing instead of guessing.
    """
    try:
        from sherlock_project.ai_engine import pass_one_contract_hash

        return pass_one_contract_hash()
    except Exception:
        return None


def sort_extractions(
    records: Iterable[SiteExtractionRecord],
) -> list[SiteExtractionRecord]:
    """Most productive site first, never-analysed last.

    NOT alphabetical, and that is the whole reason this panel is worth having
    separately from ACCOUNTS. Both ends of this order are the interesting ones:
    the best extractions are what a prompt edit is judged on, and the empty tail
    is what "should I change model" is judged on. Alphabetical buries both in the
    middle of 149 rows.

    Three keys, in order. Fact count descending is the ranking. Then analysed
    before never-analysed, so a page the model READ and found nothing on sorts
    above one nobody ever asked about -- they both show zero facts and they are
    not the same claim. Then name, so the order is stable across reloads rather
    than depending on however the rows came back.
    """
    return sorted(
        records,
        key=lambda record: (
            -record.fact_count,
            not record.analysed,
            record.site_name.lower(),
        ),
    )


def extraction_summary(records: Sequence[SiteExtractionRecord]) -> Text:
    """The distribution, in three numbers, under the list.

    This is the "should I change model" answer, and it is the one thing a
    per-site view cannot give: 8 with facts out of 149 is a judgement about the
    MODEL, while any single extraction is only a judgement about one page.

    `never analysed` is kept apart from `empty` for the reason the whole tool
    exists: a page the model read and found nothing on is a result, and a page
    nobody asked about is a gap. Summing them would report a gap as a finding.
    """
    if not records:
        return Text("")

    with_facts = sum(1 for record in records if record.fact_count)
    analysed = sum(1 for record in records if record.analysed)
    empty = analysed - with_facts
    never = len(records) - analysed

    text = Text()
    text.append(f"{with_facts} with facts", style="green" if with_facts else "dim")
    text.append(f"\n{empty} empty", style="dim")
    if never:
        # Only when there are any. A permanent `0 never analysed` would be a
        # line that says nothing on every run where analysis was on throughout.
        text.append(f"\n{never} never analysed", style="yellow")
    return text


def extraction_detail(
    record: SiteExtractionRecord | None,
    *,
    contract_hash: str | None = None,
) -> Text:
    """One site's whole story: where it came from, what it found, and why."""
    if record is None:
        return Text(
            "Select a site to read its extraction.",
            style="dim italic",
        )

    text = Text()
    text.append(record.site_name, style="bold cyan")
    text.append("\n")
    text.append_text(extraction_provenance(record, contract_hash))
    text.append("\n\n")
    text.append_text(extraction_facts(record))
    reasoning = extraction_reasoning(record)
    if reasoning.plain:
        text.append("\n\n")
        text.append_text(reasoning)
    return text


def _row(label: str, value: str, *, style: str = "", muted: bool = False) -> Text:
    """One provenance line: fixed label column, value takes what is left."""
    line = Text()
    # The label is always dim: it is the caption, and the value is the content.
    # Only the value answers to `muted`, which marks a value that is absent or
    # incidental rather than one worth reading.
    line.append(f"{label:<{PROVENANCE_LABEL_WIDTH}}", style="dim")
    line.append(value, style="dim" if muted else style)
    return line


def extraction_provenance(
    record: SiteExtractionRecord,
    contract_hash: str | None = None,
) -> Text:
    """Which model, when, and whether a re-scan would redo it.

    The staleness line is the useful half and it is a COMPARISON, never the hash
    itself: `ai_extraction_contract_hash` is 64 hex characters that answer
    nothing on sight. What someone wants to know is whether the prompt or schema
    has changed since this was written, because that is exactly what decides
    whether the next `--ai` run replaces it.
    """
    text = Text()

    model = record.ai_extraction_model
    text.append_text(
        _row(
            "model",
            model_name(model, MODEL_LINE_WIDTH) if model else "not recorded",
            style="cyan" if model else "",
            muted=not model,
        )
    )

    if record.scanned_at:
        text.append("\n")
        text.append_text(_row("scanned", record.scanned_at, muted=True))

    stored = record.ai_extraction_contract_hash
    if stored and contract_hash:
        fresh = stored == contract_hash
        text.append("\n")
        text.append_text(
            _row(
                "contract",
                "current" if fresh else "superseded — a re-scan will redo it",
                style="green" if fresh else "yellow",
            )
        )
    return text


def extraction_facts(record: SiteExtractionRecord) -> Text:
    """The extraction, in the field/value shape the profile panel uses.

    One vocabulary for one kind of content: these are the same facts Pass 2
    merges into the profile, so they are drawn the way the profile draws them
    rather than as raw JSON. The JSON was already available in a verbose scan log
    and was not what anyone wanted to read.
    """
    text = Text()
    text.append("FACTS", style="bold")

    if not record.analysed:
        # Never analysed is not the same as analysed and empty, and this is the
        # branch that says which. The other is below.
        text.append("\n\n")
        text.append(
            "This site has not been analysed.\n"
            "Scan again with analysis on to extract from it.",
            style="dim italic",
        )
        return text

    facts = record.facts
    if not facts:
        text.append("\n\n")
        text.append(
            "The model read this page and found nothing about the owner.\n"
            "That is a result, not a failure — most pages hold no facts.",
            style="dim italic",
        )
        return text

    text.append(f"   {record.fact_count} from {len(facts)} fields", style="dim")
    for key, values in facts.items():
        # The field once per group, values under it -- the grouping
        # `render_profile` uses, so a reader moving between the two surfaces is
        # not learning a second layout for the same thing. The indent on
        # continuation lines is what shows a group, so no blank line between
        # groups is needed and vertical space in a panel is scarce.
        label = key.replace("_", " ")
        if len(label) > FIELD_LABEL_WIDTH - 1:
            label = label[: FIELD_LABEL_WIDTH - 2] + "…"
        for index, value in enumerate(values):
            text.append("\n")
            text.append(
                f"{label if index == 0 else '':<{FIELD_LABEL_WIDTH}}",
                style="",
            )
            # Text(), never markup: this is model output about a scanned page,
            # and Rich would read brackets in it as a style tag.
            text.append(value, style="cyan")
    return text


def extraction_reasoning(record: SiteExtractionRecord) -> Text:
    """How the model got there -- the developer's half of this panel.

    THREE DISTINCT EMPTY CASES, and collapsing them would make this useless
    exactly when it matters. A row written before the column existed has nothing
    recorded; a model whose native thinking cannot be turned off was sent the
    variant prompt and had no field to fill; and a model that was asked and
    returned nothing is a prompt problem worth seeing. Only the first two are
    benign, and none of them is "the model did not think".
    """
    if not record.analysed:
        # Nothing was asked, so there is no reasoning to be missing. Returned
        # empty so the caller omits the heading rather than printing one over an
        # explanation of an absence that is already explained above.
        return Text()

    text = Text()
    text.append("REASONING", style="bold")

    reasoning = record.ai_extraction_reasoning
    if not reasoning:
        text.append("\n")
        text.append(
            "Not recorded. Either this extraction predates reasoning being "
            "stored, or the model reasons natively and was sent the prompt "
            "variant that has no reasoning field.",
            style="dim italic",
        )
        return text

    # Split on the separator the prompt asks the model to use. The field is
    # specified as "one short clause per owner-evidence line", and the model
    # returns them joined with semicolons -- so as one paragraph it is a wall of
    # text, and one clause per line is the shape it was written in. Falls back to
    # the whole string when there is nothing to split on, because a model that
    # ignored the separator still said something.
    clauses = [part.strip() for part in reasoning.split(";") if part.strip()]
    text.append(
        f"   {len(clauses)} decisions, in page order"
        if len(clauses) > 1
        else "   in page order",
        style="dim",
    )
    for clause in clauses or [reasoning]:
        text.append("\n")
        # `skip:` lines are dimmed. On a busy page most clauses are skips, and at
        # equal weight they bury the handful that produced a fact -- which is the
        # half a reader is checking against FACTS above. Still shown in full: WHY
        # something was skipped is the question a prompt edit usually asks.
        skipped = clause.lower().startswith("skip")
        text.append(clause, style="dim" if skipped else "")
    return text


def facts_cell(record: SiteExtractionRecord | None) -> Text:
    """What Pass 1 made of one site, for a narrow table column.

    THREE STATES, NOT TWO, and the distinction is the same one the whole tool is
    built on. `—` means nobody has asked the model about this page; `0` means it
    was asked and the page held nothing. Rendering both as blank would turn "not
    analysed" into "nothing there", which is the absence-of-evidence error the
    SITES counters refuse to make.

    Shared by the ACCOUNTS list and the extraction panel's own list, so one site
    cannot report differently depending on which one you are looking at.
    """
    if record is None or not record.analysed:
        return Text("—", style="dim")
    count = record.fact_count
    if not count:
        # Dim, but a real zero. It is an answer.
        return Text("0", style="dim")
    return Text(str(count), style="green")
