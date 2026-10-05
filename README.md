# Hybrid RULERS reproducibility package

This repository contains the analysis code, frozen protocols, manuscript source,
figures, and audit outputs supporting the Hybrid RULERS metric review. It is an
offline research package for reproducibility and review; it is not a clinical
decision-support system and it does not contain the source audio archive.

## Important data-use boundary

The Pitt Cookie Theft and Lu materials are from DementiaBank/TalkBank and are
subject to their access and data-use rules. Do not publish or upload audio,
original CHAT files, unrestricted transcripts, API request/response logs,
participant-level tables, or identifying file paths. Keep those inputs and any
provider credentials outside this repository. The TalkBank Ground Rules and
the applicable authorization must be followed for every reuse.

This directory is a public-release candidate. Review every included file and
confirm the repository and data-use terms with all authors before publishing.
The full internal audit archive is intentionally kept outside this directory.

## Layout

- `stage_a/`: offline feature, model, reanalysis, and audit scripts.
- `stage_b/`: fixed-transcript repeatability protocol, analysis code, and audit reports.
- `lu/`: exploratory Lu external-evaluation code and provenance notes.
- `source_tables/`: manuscript-facing summary tables; inspect every table before release.
- `manuscript/`: manuscript source and figures used to produce the paper.
- `PACKAGE_FILE_HASHES.csv`: hash manifest for the internal package.

## Environment

Python 3.10 or newer is recommended. Install the numerical dependencies with:

```text
python -m pip install -r requirements.txt
```

The scripts consume locally prepared, authorized inputs and frozen intermediate
files. Several historical entry points retain absolute paths from the original
workstation; update those paths or provide a local adapter before attempting a
new run. The package does not call an external API during offline analyses.
The API runner is retained for provenance only and must never be run with
restricted data unless the applicable data-use and provider-retention
requirements have been independently verified.

## Reproduction principles

1. Treat the stored primary results as frozen; do not tune thresholds or
   features from test-set outcomes.
2. Keep sensitivity analyses separate from the primary results.
3. Record the data-access basis, software versions, protocol hash, and input
   hashes for any rerun.
4. Do not infer clinical calibration, clinical utility, or validated safety
   from these research metrics.

## License and citation

No license is asserted by this package yet. Add a license only after all
authors and the data provider requirements have been confirmed. Cite the
associated manuscript and the DementiaBank/TalkBank resource as required by
their terms.
