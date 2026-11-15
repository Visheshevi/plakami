# ADR-0015: Owner accepts the v1 gap to the SPEC §6 targets

## Context

SPEC §10 says T4 ends when the §6 targets are met "or an ADR records the accepted gap".

The release candidate is `runs/yolox_tiny_640_v1` (ADR-0013) with 2×2 tiled detection. Its scores are on the audited labels (ADR-0014), with plates under 12 px ignored, at plate confidence 0.2:

| Metric | Value | Target | Met |
|---|---|---|---|
| Plate recall, test | 0.947 | ≥ 0.95 | no (≈4 plates short) |
| Small-plate recall (12–32 px), test | 0.849 | ≥ 0.85 | no (1 plate short) |
| Plate recall, holdout | 0.900 | ≥ 0.95 | no |
| Vehicle recall, test | 0.741 | ≥ 0.90 | no |
| Plate false positives per image, test | 0.418 | ≤ 1.0 | yes |

Without tiling, the same model scores 0.904 on test plates and 0.694 on small plates. Most remaining holdout misses are small, blurry, distant plates. Most vehicle misses are Open Images label issues (T3 report).

## Decision

On 2026-11-14 the owner accepted this model for v1 ("I am good with the model for now"). The targets are **not** lowered. The gap is recorded here and will be stated in the model card and release notes.

- **Release candidate.** v1 checkpoint SHA-256 `48086beb…`. The recommended usage is whole image plus 2×2 tiles (`detect_tiled`, overlap 0.2) for single photos.
- **Thresholds.** The per-class recommended thresholds in `release.json` are chosen on `val` (SPEC §6) during T5/T6, never on test or holdout.
- **Possible future work** (not scheduled):
  - a val-tuned plate threshold;
  - an 800 px model for small or distant plates (ADR-0013 step C);
  - cleaning the vehicle labels;
  - the vehicle-target question.

## Consequences

- The release notes and `MODEL_CARD.md` list the gap, the tiled-usage recommendation, and its cost of 5 model passes per photo.
- Still pending owner approval:
  - SPEC §6 wording on unreadable plates (ADR-0013);
  - the label-audit convention (ADR-0014);
  - the earlier ADR-0007 and ADR-0010 wording.
