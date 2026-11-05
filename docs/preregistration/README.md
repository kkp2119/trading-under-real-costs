# Pre-registration records

These are the original rule files, written before any result they govern was computed. They are in Korean, and the first line of each Korean file says "(Korean)". The English one-page summaries are in `docs/`.

| Study | Files |
|---|---|
| 12 | `SPEC.md` (rules, grid, verdict), `SPEC_AMENDMENTS_draft_*.md` (recorded changes), `CODE_FREEZE_MANIFEST.tsv` (SHA-256 of the frozen code), `FREEZE_TEST_REPORT.txt` |
| 13–14 | `SPEC.md`, `SPEC_AMENDMENTS.md`, `CODE_FREEZE_MANIFEST.tsv`, `FREEZE_TEST_REPORT.txt`, `dev_prediction_scoring.md`; Study 14: `SPEC14.md`, `CODE_FREEZE_MANIFEST14.tsv`, `FREEZE_TEST_REPORT14.txt`, `holdout_prediction_scoring14.md` |
| 15 | `SPLIT15.md` (data split), `FEATURES15.md`, `EXEC15_DECLARATION.md`, `TEST15_DECLARATION.md` (sealed test rule), `TEST15_PREDICTIONS.md` and `test15_prediction_scoring.md`, `DEVIATIONS15.md` (deviations log) |
| 16 | `GATE16.md`, `QUEUE16.md`, `ROBUST16.md`, `QTY16.md` with their `.sha256` files, and `DEVIATIONS16.md` |

**Changes from the originals.** The files in this public copy differ from the originals in three ways:
- The "(Korean)" first line was added.
- Local folder paths were replaced with `$DATA_DIR/`, and the name of the local WRDS credential file was replaced with a generic description.
- In `DEVIATIONS16.md`, the quote and trade values of one stock-second used in a hand check were removed, because they are licensed TAQ data.

Because of these changes, the SHA-256 values recorded in the `.sha256` files and manifests refer to the unmodified originals in the private research repository, not to these copies. Nothing else was changed.
