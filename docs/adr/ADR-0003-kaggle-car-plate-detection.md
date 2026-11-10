# ADR-0003: Add the Kaggle car-plate dataset as a CC0 source

- **Status:** Accepted
- **Date:** 2026-11-02

## Context

The allowlist in SPEC §4.1 includes `andrewmvd/car-plate-detection` as CC0, with 433 plate-labeled images. Kaggle's dataset card identifies CC0 1.0, PASCAL VOC annotations, and credits Make ML as the source. Its public API redirects anonymously to the archive endpoint; no Kaggle account or credentials are required by the implementation. The archive is approximately 213 MB.

## Decision

Add an adapter for dataset version 1. Record the Kaggle dataset page, uploader, and CC0 1.0 URL in each manifest entry. Parse only the recognized plate labels and reject unsupported labels, malformed boxes, unsafe archive paths, oversized archives, and invalid image files. Require an explicit `MAX_IMAGES` cap; this caps processed images, not the archive download. Do not read Kaggle credentials.

## Consequences

The dataset can supplement plate examples with a small, reproducible source. It contains plate boxes only and no vehicle annotations; its images must not be interpreted as verified vehicle-negative examples when training a two-class detector. The Kaggle card's CC0 statement is the provenance/licence basis used here; the upstream acknowledgement is retained in this ADR and README. A real archive run using `MAX_IMAGES=100` produced 91 usable images after removing 9 pHash duplicates, with 108 plate boxes and no build failures. The source's actual Pascal VOC class spelling was `licence`; the parser normalizes this to the allowlisted plate class. This small run validates format compatibility, not the dataset's broader legal provenance beyond the Kaggle CC0 declaration.
