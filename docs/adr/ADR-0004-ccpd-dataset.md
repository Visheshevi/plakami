# ADR-0004: Add CCPD as a capped plate-only data source

- **Status:** Accepted
- **Date:** 2026-11-02

## Context

The project allowlist names CCPD (2019/2020) as an MIT source. The official `detectRecog/CCPD` README states that CCPD is open-source under MIT, describes 300k+ images, and explains that bounding boxes are embedded in image filenames. Its filename sample encodes top-left and bottom-right box coordinates. The official download is a Google Drive tar.xz link; its confirmation page identifies `CCPD2019.tar.xz` as approximately 12 GB. It is a bulk archive rather than a selective per-image download. The CCPD MIT license file is in the official repository, and the dataset-specific README also states the dataset licence.

## Decision

Implement local processing from an already-extracted official CCPD directory. Do not download the full archive to the Mac: it had about 29 GiB free and the source archive is approximately 12 GB before sample export. Add an optional Colab helper that downloads only if the final HTTP response exposes an exact size below 16 GiB and the free runtime disk has that size plus an 8 GiB reserve. If size or disk checks fail, abort before reading archive bytes. Stream the tar.xz and retain at most 100 deterministically selected JPEGs; bundle only that bounded sample for manual transfer back to the repo. No Colab session is started automatically. Record the official CCPD project page as provenance, `CCPD authors` as the available author attribution, and the official repository MIT license URL. Parse only valid CCPD JPEG filename boxes; skip files that do not match the annotation format. Mark CCPD manifest/COCO rows as plate-only. In the combined dataset, cap CCPD to at most 30% of all plate boxes, as confirmed by the owner.

## Consequences

The adapter can be tested without acquiring the bulk archive, and a user-provided/extracted official directory can be processed with a strict image cap. CCPD's Chinese parking-lot imagery supplements plate examples but has no vehicle boxes; these images must not be used as vehicle-negative examples. The combined output tracks per-image annotation scope, globally deduplicates source records, reapplies deterministic splits, and reports the realized CCPD share. The owner reported one successful Colab run that downloaded the archive and selected 100 images, followed by a retry that received HTML instead of the archive. The helper now follows a bounded number of recognized Drive confirmation forms, rejects non-Google form targets, and classifies likely quota/access/virus-scan pages without printing hidden fields. All local tests use synthetic responses; the downloaded sample bundle has not yet been transferred to the workspace.

## Addendum (2026-11-09): CCPD2020 sample used while CCPD2019 is quota-blocked

The CCPD2019 Drive file returned a download-quota page on two consecutive attempts. The same official README (`detectRecog/CCPD`, "UPdate on 16/09/2020") links a separate Drive file for CCPD2020 / CCPD-Green (ID `1m8w1kFxnCEiqz_-t2vTcgrgqNIv986PR`), under the same "open-source under MIT license" statement; SPEC §4.1 allowlists "CCPD (2019/2020)". We downloaded `CCPD2020.zip` (907,711,344 bytes, SHA-256 `7ce77266d2fa216e1903df47be800b25fa1f408b985920dbb5e15bc6cfd72271`; 11,776 JPEGs in `ccpd_green/{train,val,test}`) to `data/raw/ccpd/` and extracted the 100 lowest-SHA-256 paths to `data/raw/ccpd/extracted/`. Filenames use the same box encoding (8-character plate field), so the existing parser works unchanged.

Consequence: the current CCPD sample is **green new-energy plates only**, a narrower style than CCPD2019's blue plates. This is acceptable for pipeline validation; the training dataset should add CCPD2019 once its quota clears. The download used a one-off script outside the repo; a reproducible in-repo downloader is a follow-up.
