# ADR-0013: Small plates — readable-size rule, tiled detection and v1 training

## Context

The v0 640 model (`runs/yolox_tiny_640_v0`) finds 96% of test plates wider than 64 px but
62% of 16–24 px plates and almost none under 12 px (holdout: 0 of 31). SPEC §6 asks for
small-plate recall ≥ 0.85 (< 32 px) and plate recall ≥ 0.95 on test and holdout.

Owner decisions (2026-11-13):

- Keep the 640 model; drop 416 (416 found fewer plates at every size).
- Apps process **one photo at a time**; live camera video is not planned. Running the
  model several times per photo is acceptable.
- Plates too small to be read in the photo are **ignored**.
- Run a small-plate retraining (step B) on the Mac.

## Decision

1. **Readable-size rule.** A plate under **12 px wide in the stored image** is unreadable.
   - Evaluation ignores it: it is not a miss, and a detection on it is not a false positive.
   - Training drops it (instead of v0's "short side under 6 px at 640", which also dropped
     readable 20–30 px wide plates).
   - The SPEC §6 wording change (below) is proposed to the owner, not applied.
2. **Tiled detection (step A).** Run the model on the whole photo plus overlapping tiles and
   merge the boxes. The tiling and merge rule lives once in `src/platedet/postprocess.py`.
   Tile layout and thresholds are chosen on `val` only.
3. **v1 training (step B), `configs/train/yolox_tiny_640_v1.yaml`.** Same model, input size
   and schedule as v0. Changes:
   - plates kept by the readable-size rule (535 more training plates: 117 dropped vs 652);
   - multi-scale sizes 544–832 px (v0: 480–800). 896 px would overflow 16 GB memory
     (ADR-0012);
   - mosaic zoom 0.5–2.0 (v0: 0.5–1.5), so training also sees plates at the scale tiles show.
4. **Step C (800 px input)** only if A and B fall short.

## Proposed SPEC §6 wording (owner to approve)

> Plates narrower than 12 px in the evaluated image are unreadable and are excluded from
> recall and false-positive counts. "Small plates" means 12–32 px wide.

## Consequences

- Recall numbers from v0 reports are not directly comparable; reports show both rules
  until the SPEC is updated.
- Our evaluation images are capped at 1,024 px; phone photos are 3,000–4,000 px, so tiling
  should help more in production than measured here.
- Tiling costs about 5× the compute of one pass per photo.
