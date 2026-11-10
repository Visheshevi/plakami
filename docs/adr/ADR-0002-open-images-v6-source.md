# ADR-0002: Use documented Open Images V6 endpoints

- **Status:** Accepted
- **Date:** 2026-11-01

## Context

The spec names Open Images V7, but the official Open Images download and description pages available during T1 identify V6 as the latest version and document its bounding-box annotations, image metadata, and class labels. The official pages also warn that image licences are listed as CC BY 2.0 without a warranty and should be verified individually.

## Decision

Implement the first ingestion source against the official Open Images V6 class/annotation CSVs and V6-era image metadata endpoints. Do not edit the spec. Accept only metadata rows with the exact CC BY 2.0 licence URL, a non-empty author, and an HTTP(S) source landing URL; exclude rows that fail these checks. Keep an explicit `MAX_IMAGES` cap for each invocation.

## Consequences

The implementation is reproducible against a documented source rather than an unverified V7 claim. The generated manifest records source landing URL, author, licence URL, and content hashes. The upstream disclaimer means metadata validation is not a legal verification of each image; before training or release, the owner must review whether this source meets the project's commercial-use standard. Switching to verified V7 requires updating this ADR and the implementation after authoritative documentation is found.