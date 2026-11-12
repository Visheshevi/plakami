# ADR-0006: Download Open Images image files from the official CVDF host

- **Status:** Accepted (owner approved 2026-11-10)
- **Date:** 2026-11-11

## Context

The Open Images builder downloaded each image from its Flickr `OriginalURL`. In the first full build, Flickr refused 2,359 of 2,500 requests with HTTP 429. Retry-with-backoff (commit `6340da9`) cut the refusals, but no new images arrived in two hours. About 15–20% of Flickr originals also return 404/410 because the photos were deleted.

The official Open Images V6 download page says images can be downloaded from CVDF, and links Google's `downloader.py` (Apache-2.0), which reads from the public bucket `open-images-dataset` by `<split>/<image_id>`. A HEAD check of one image from our sample returned HTTP 200, `image/jpeg`, 166 KB, against 1.0 MB for the Flickr original. The CVDF copies are therefore re-encoded or resized versions of the same images.

## Decision

Add `image_host` (`flickr` | `cvdf`) to the Open Images data config. The project config uses `cvdf`, downloading `https://open-images-dataset.s3.amazonaws.com/<split>/<image_id>.jpg` over HTTPS with the existing retry logic. We do not vendor or run `downloader.py`. Licence and attribution handling is unchanged: every row still needs the exact CC BY 2.0 licence URL, a non-empty author and the Flickr landing URL from the official metadata CSV, and the manifest records those, not the CVDF URL. Open Images annotations are normalised coordinates, so boxes are unaffected by the image size.

## Consequences

Downloads are faster and smaller (an estimated 1.3 GB for ~7,600 images instead of ~15 GB), and images deleted from Flickr remain available. Manifest `sha256`/`phash` values describe the CVDF files, so they differ from earlier Flickr-sourced samples; the Open Images processed folder must be rebuilt rather than mixed. CVDF images may be lower resolution than the Flickr originals, which can make small plates smaller in pixels; `stats.json` plate-size figures will show this. The per-image licence disclaimer in ADR-0002 still applies.
