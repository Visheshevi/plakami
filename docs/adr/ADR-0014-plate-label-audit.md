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

## Review record (2026-11-14)

- 5 reviewer agents, 17 sheets each, same criteria. Results:
  - test: 1,283 ok, 27 loose, 40 not_plate;
  - holdout: 311 ok, 2 loose, 20 not_plate.
- I re-checked 2 random sheets from the all-ok CCPD batch (40 boxes): all ok.
- Owner spot check: 50 random decisions (25 flagged, 25 ok, seed 20261114). The owner agreed with all 50.
  - On #3 (test/659, loose) the owner noted the box covers half the plate, which is still loose.
  - On #22 (test/1354) the owner noted an unlabelled plate on another car. Unlabelled plates are out of scope.
- Most not_plate labels are deliberately blanked plates (Open Images anonymisation) or dealer, show or novelty inserts with no registration. Whether these count as plates is an open owner decision, as is redrawing versus ignoring loose boxes.

With flagged labels ignored, v1 + 2×2 tiles scores:

| Metric | Audited | Unaudited | Target |
|---|---|---|---|
| Test plate recall | 0.947 | 0.933 | 0.95 |
| Test small-plate recall | 0.849 | 0.841 | 0.85 |
| Holdout plate recall | 0.900 | 0.886 | 0.95 |

## Owner rules on flagged labels (2026-11-14)

- **Blanked-out or covered plates count as plates.** This covers deliberate anonymisation (fills, blur, bars) and blank covers.
- **Dealer, show, novelty and taxi inserts in a plate holder count as plates.**
- **Result.** 52 `not_plate` labels were re-checked by eye for box fit:
  - 49 became `ok`;
  - 2 became `loose` (holdout/110, test/100);
  - 1 stayed `not_plate` (test/1589: dark empty recess, no plate or insert visible).

  Their notes record the change.
- **Remaining `not_plate` (9):** a badge on a body panel, a car body edge, a dark tailgate recess, a bus fine notice, painted artwork, a tram sign, a bumper corner, a house-number sign, and test/1589.
- **Final counts:**
  - test: 1,316 ok, 28 loose, 6 not_plate;
  - holdout: 327 ok, 3 loose, 3 not_plate.
- **Still open:** redraw or ignore the 31 loose boxes (sheets in `data/processed/label_audit/loose/`).

v1 + 2×2 tiles, audited with these rules:

| Metric | Audited | Target |
|---|---|---|
| Test plate recall | 0.948 | 0.95 |
| Test small-plate recall | 0.849 | 0.85 |
| Holdout plate recall | 0.902 | 0.95 |

## Consequences

- Ignoring flagged labels shrinks the denominator. Reports show the flagged count per split,
  so the effect is visible.
- Redrawing (if the owner chooses it) would replace ignoring for `loose` labels; `not_plate`
  labels would be removed either way.
