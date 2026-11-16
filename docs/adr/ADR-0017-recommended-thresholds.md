# ADR-0017: Recommended thresholds for v1

## Context

SPEC §6 asks for a recommended operating threshold per class, chosen from the precision/recall curve and written into `release.json`. The SPEC §6 targets themselves are measured at fixed thresholds: 0.2 for plates and 0.3 for vehicles.

The release candidate is v1 with 2×2 tiles. These are its `val` results (unreadable plates ignored):

| Plate threshold | Recall | Small plates | False plates / image |
|---|---|---|---|
| 0.30 | 0.929 | 0.780 | 0.29 |
| 0.20 | 0.933 | 0.796 | 0.36 |
| 0.10 | 0.938 | 0.804 | 0.50 |
| 0.05 | 0.941 | 0.808 | 0.68 |

| Vehicle threshold | Recall | False vehicles / image |
|---|---|---|
| 0.20 | 0.801 | 1.91 |
| 0.30 | 0.770 | 1.45 |
| 0.40 | 0.734 | 1.19 |

No threshold reaches 0.95 plate recall on `val`. Note that `val` labels are unaudited, unlike test and holdout (ADR-0014).

## Decision

On 2026-11-15 the owner approved:

- **Plates: 0.10.** For privacy, a missed plate costs more than an extra blurred patch. False plates stay under SPEC's 1.0 per image.
- **Vehicles: 0.30** (the SPEC default). Vehicles are used to check that a photo contains a car, and lower thresholds add many false boxes.

These are chosen on `val` only. They are recorded in `configs/release/v1.yaml` and go into `release.json` in T6.

## Consequences

- Reports keep measuring the SPEC §6 targets at 0.2 and 0.3, so results stay comparable over time.
- The recommended thresholds are defaults; apps can set their own per class.
