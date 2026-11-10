# ADR-0001: Keep the initial scaffold dependency- and asset-minimal

- **Status:** Accepted for T0
- **Date:** 2026-11-01

## Context

T0 needs reproducible project tooling, linting, and tests, while data sources and model/export dependencies must be reviewed for licence compatibility before use. The spec calls for a small smoke dataset, but no specific image assets have yet been selected and verified.

## Decision

The initial project declares no runtime dependencies. The development group contains only pytest and Ruff; the build backend is Hatchling. The initial test is asset-free. Training, data, export, and parity dependencies and smoke images are deferred to their relevant steps after licence review.

## Consequences

The scaffold can be installed and checked without downloading datasets or model packages. CI verifies package installation, lint, and the asset-free test. The scaffold does not yet verify a training or export path; those checks belong to later steps.