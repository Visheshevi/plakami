# ADR-0005: Sample Open Images plate images first

- **Status:** Accepted
- **Date:** 2026-11-09

## Context

The Open Images builder sampled uniformly from every image with a plate or vehicle box. Vehicle images far outnumber plate images, so a 100-image sample held 1 plate box against 186 vehicle boxes. The owner confirmed the model will be used worldwide and that no owner photos are available. That makes Open Images (global Flickr photos) the main source of non-Chinese plates; CCPD is single-region and capped at 30%, and Kaggle is small.

## Decision

Add `plate_image_fraction` to the Open Images data config. Up to that share of the `MAX_IMAGES` cap is filled with images containing at least one plate box; vehicle-only images fill the rest, and any shortfall on either side is filled from the other. Both groups keep the existing deterministic SHA-256 ordering. The project config sets `0.8`; the code default is `0.0` (uniform-like behaviour). `stats.json` reports the selected plate and vehicle-only counts and the images that ended up with plates.

## Consequences

Plate examples now dominate Open Images samples, while vehicle boxes still come from both groups, since plate images usually also contain vehicle boxes. The sample is no longer representative of Open Images' natural class balance, which is acceptable for training. Evaluation splits still come from the same hash split, so `test` is plate-heavy too; this matches the plate-recall priority in SPEC §6. The fraction is a tunable experiment setting, not a fixed ratio.
