# ADR-0018: v1.0.0 release package, MIT licence, built locally first

## Context

SPEC §8 lists the release assets and says releases are created by CI from a tag. SPEC §11.3
left the licence for the code and weights open. There is no GitHub repository or remote yet.

On 2026-11-15 the owner decided:

- build the release package locally now, and push to GitHub as a later step;
- **MIT** for this repository's code and the released weights (Apache-2.0 was proposed;
  the owner chose MIT);
- the mobile integration `README.md` is part of T6.

## Decision

1. **`make release`** builds `runs/<run_id>/release/v<version>/` and `platedet-v<version>.zip`
   from `configs/release/v1.yaml`. Contents:
   - the SPEC §8 assets, FP16 only (ADR-0016);
   - `README.md` (integration guide), `LICENSE` (MIT), `THIRD_PARTY_NOTICES.md` and
     `licenses/YOLOX-Apache-2.0.txt`.
2. **Same files as evaluated.** The builder refuses to package model files whose SHA-256
   differs from the evaluation summary or the parity report, and refuses test vectors built
   for another release config.
3. **Metrics in `release.json` and `MODEL_CARD.md`:**
   - the SPEC §6 targets at 0.2 / 0.3 (audited test and holdout, 2×2 tiles), as reported
     before;
   - the same sets at the recommended thresholds (ADR-0017), computed from the saved
     predictions. This is reporting only; nothing is tuned on test or holdout.
4. **`make release-check`** works like an outside user. It copies the package to an empty
   folder, verifies `SHA256SUMS` and the hashes in `release.json`, and reproduces every test
   vector (ONNX and FP16, single pass and tiles) using only the packaged `postprocess.py`
   and model files. It also fails while the copyright placeholder is still in `LICENSE`.
5. **The README's algorithm is tested.** A unit test re-implements the README's decoding
   and NMS steps without `postprocess.py` and compares them with the reference. A one-off
   check also reproduced all 5 test vectors with that independent version.
6. **Licences.**
   - MIT covers this repository and the weights.
   - YOLOX is Apache-2.0, which allows a derivative under other terms if its licence text and
     notices are kept. They ship in `THIRD_PARTY_NOTICES.md` and `licenses/`.
   - Training images (CC BY 2.0, MIT, CC0) are credited in `ATTRIBUTION.md`, generated
     from the dataset manifest.
7. **CI publishing waits for the repository.** SPEC §8 (CI release from a tag) and the T6
   "done when" check (download from a release URL) need the GitHub repository. That is the
   next step after the owner approves the package.

## Name

The public name is **Plakami** (owner, 2026-11-15): Turkish *plaka* (licence plate) and Japanese *mi* (見, to see). Releases are `plakami-v<version>.zip`. The internal Python package stays `platedet`; users never import it.

## Consequences

- `README.md` at the repository root is now the integration guide. The old T1 developer
  notes moved to `docs/development.md`.
- `export.json`'s top-level `onnx_sha256` (0c26…) is from an earlier export of the same
  checkpoint. The current `model.onnx` (25aeb…) is the file that was evaluated,
  parity-checked and converted to TFLite. A PyTorch comparison on 2026-11-15 confirmed it
  matches the v1 checkpoint (max error 8e-6). `release.json` records the actual hashes.
- The Kotlin and Swift examples in the README have not been compiled. They are marked as
  such, and ports are checked with the test vectors.
- **Proposed SPEC wording** for the owner:
  - §11.3: licence decided, MIT;
  - §8: INT8 is not shipped in v1;
  - §8: the package also includes `README.md`, `LICENSE` and the third-party notices.
