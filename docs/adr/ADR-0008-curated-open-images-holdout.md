# ADR-0008: Curated holdout from the Open Images test split

- **Status:** Accepted (owner chose this option on 2026-11-12)
- **Date:** 2026-11-12

## Context

SPEC §4.4 and §6 describe a `holdout` set of real-world photos, ideally the owner's own, with a plate-recall target of ≥ 0.95. The owner can't provide photos (2026-11-09). Our `test` split is useful, but its Open Images labels are incomplete and sometimes wrong, so plate recall on it is noisy.

The first attempt took unused plate images from the Open Images train and validation splits. That found only 12, because the training sample (7,600 images, 80% plate-first) already uses almost every one of the roughly 6,090 plate images in those splits.

## Decision

Build the holdout from the **official Open Images V6 test split** (boxes `v5/test-annotations-bbox.csv`, metadata `2018_04/test/test-images-with-rotation.csv`, images from CVDF). The training build never reads this split.
- **Same source and licence rules.** This is the same allowlisted source with the same CC BY 2.0 licence and attribution checks, so no allowlist change is needed.
- **No overlap with training.** The config pins `source_splits: [test]`, and `prepare` refuses any overlap with the training splits.
- **No near-duplicates.** Candidates are removed if they are pHash near-duplicates (distance ≤ 4) of anything in the combined dataset.

**Process:**
1. `make holdout-prepare` downloads 300 candidates, chosen in a fixed order with the salted hash `holdout:<id>`, and draws numbered review sheets with the plate boxes.
2. Every candidate is reviewed by eye and either accepted or rejected as a whole image. Boxes are never added or edited by hand.
3. A candidate is accepted only if:
   - every box is on a real plate and lines up with it;
   - no box covers several plates at once;
   - no clearly recognisable plate is unlabelled.

   Rejected candidates get a reason.
4. The decisions are committed as `configs/holdout/review.csv`. It holds only image IDs, decisions and reasons, with no images or boxes, so the holdout can be rebuilt exactly.
5. `make holdout-finalize` writes the set to `data/processed/holdout/` (git-ignored).

**Scoring:**
- Only plates are scored (`annotated_classes = ["license_plate"]`). Vehicle labels were not reviewed.
- The evaluation runner reads the holdout from `holdout_dir`.

## Consequences

- **Cleaner but easier.** The holdout is cleaner than `test`, but it is still Flickr photos, not phone photos from the deployment setting. Rejecting images with label problems may also bias it towards easier scenes, with fewer and larger plates. The stats report its plate count and small-plate count so that bias is visible.
- **Evaluation only.** It must never be used for training, threshold tuning or INT8 calibration.
- **Reviewer coverage.** The first review was split across parallel reviewer agents with identical written criteria, and a sample of their decisions was re-checked.

## Result (2026-11-12)

- **Review:** 300 candidates were reviewed. 4 parallel reviewers used the criteria above. I re-checked 24 random accepts at full size and overruled 2: one had a fully whited-out plate (an edited image), and one had a box on an unclear sign at the frame edge.
- **Accepted:** 244 images with 364 plate boxes. 98 boxes (27%) are under 32 px wide.
- **Rejected:** 56 images:
  - 34 box_not_on_plate, mostly motor-show and dealer display inserts showing a model name, which Open Images labels as plates;
  - 13 unclear;
  - 8 unlabelled_plate;
  - 1 plate_redacted.
- **Residual error:** going by the spot check, roughly 5–10% of accepts may still be borderline.
