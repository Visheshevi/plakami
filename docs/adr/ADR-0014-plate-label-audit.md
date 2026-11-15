# ADR-0014: Plate label audit for test and holdout

## Context

With v1 and 2×2 tiles (ADR-0013), test plate recall is 0.933 against a 0.95 target. Of the
90 missed test plates, 57 have a confident detection that overlaps the label by less than 0.5.
The miss gallery shows many loose, shifted or empty-holder labels from Open Images. Scores
therefore understate the model by an unknown amount.

Owner decisions (2026-11-14):

- Audit the plate labels in `test` and `holdout`.
- Reviewer agents review every box; the owner spot-checks a random sample.
- Bad labels are reported to the owner, who decides later whether to redraw them or ignore
  them. Until then, reports show the audited numbers next to the unaudited ones.

## Decision

1. **What is audited.** Every plate label at least 12 px wide (ADR-0013) in `test` and
   `holdout`: 1,350 + 333 = 1,683 boxes. Unlabelled plates (which only affect the false-positive
   count, already within target), vehicle labels and `val` are out of scope.
2. **Model-blind review.** `make audit-prepare` draws numbered sheets of zoomed crops with
   the label box only. Reviewers never see model output, and every box is reviewed, not only
   the ones the model missed.
3. **Decisions:**
   - `ok`;
   - `loose` (real plate, box clearly wrong);
   - `not_plate` (no plate in the box).

   Written criteria: `data/processed/label_audit/CRITERIA.md`, copied below.
4. **Committed record.** `make audit-finalize` checks the review covers every box once and
   writes `configs/audit/plate_labels.csv` (box ID, decision, note: no boxes or images) and
   `plate_labels.meta.json` (SHA-256 of the annotation files it was made against). Scoring
   refuses the audit if the annotations change.
5. **Scoring.** `make eval` adds an *Audited* column. Flagged labels are ignored there in the
   same way as unreadable plates (not a miss; a detection on one is not a false positive).
   The default *Value* column is unchanged until the owner decides between ignoring and
   redrawing. `make eval RESCORE=1` re-scores saved predictions without rerunning the model.

## Criteria (as given to reviewers)

- `ok`: a real plate; the box fits reasonably (small slack, up to ~25% extra margin, partly
  hidden or edge-cut plates with the visible part boxed). If unsure between ok and loose, ok.
- `loose`: a real plate, but the plate would cover less than ~2/3 of the box, or the box misses
  more than ~1/4 of the plate.
- `not_plate`: no plate in or near the box (empty holder, bumper, sign, sticker, blank or
  blurred-out area). A blurry, dark or small real plate is not `not_plate`.

## Consequences

- Ignoring flagged labels shrinks the denominator. Reports show the flagged count per split,
  so the effect is visible.
- Redrawing (if the owner chooses it) would replace ignoring for `loose` labels; `not_plate`
  labels would be removed either way.
