# 2026-07-02 benchmark snapshot

These files preserve the HumanEval and LiveBench evidence used to generate the
original project teaser. They are raw historical inputs, not current schema-v5
benchmark artifacts.

## Contents

| File | Result rows | Logged calls | SHA-256 |
|---|---:|---:|---|
| `humaneval/results_product.json` | 164 | 328 | `ad2307b34657e02cbbb5529eafe926ea6786bade1ef76bdbaa5ff873dfdfeb33` |
| `livebench/results_product.json` | 128 | 234 | `94e6e0423c2103417bab5e5af79057abe5d2970903f622476f364509b95125ab` |
| `livebench/results_opus_baseline.json` | 128 | 59 new calls | `2451400dd37c55ec1a0ab9e7fd9e2011f0e7550133ec21a8374aeea6829d8a95` |
| `plot_teaser.py` | — | — | `eacd55ae67d4b879a7197d9eca49ee96e720ce808787502f2f982de076362e2c` |
| `teaser.png` | — | — | `54035e15944ccb46cc84d1093c76535e41db207017fbea4f896f051e7d96c75b` |

## Provenance

- HumanEval product results originated in commit `fcc2bf1`.
- LiveBench product results originated in commit `fc2e0c7`.
- The hybrid Opus baseline originated in commit `28964ac`.
- The teaser script and image originated in commit `e456552`.

The files do not record the full strategy snapshot, prompt contents, evaluator
configuration, code identity, dependency versions, timestamps, or usage
completeness required by the current artifact schema.

## Known limitations

- The LiveBench product file contains 128 result rows but only 106 classifier
  calls. Classifier usage is therefore absent for 22 tasks.
- The Opus baseline reused cached product outcomes and logged only 59 newly run
  calls. Its token log is not a complete standalone baseline ledger.
- Against the checked-in product file, the 73 cached product Opus task IDs and
  59 baseline task IDs overlap on five tasks and cover only 127 unique tasks.
  Summing their costs double-counts five tasks and omits one.
- `plot_teaser.py` performs that sum. With these exact files it computes an Opus
  cost of `$1.255365` and savings of about `29.6%`, while old inline comments say
  `$1.207` and roughly `27%`.
- The HumanEval always-Opus cost of `$0.97` is an estimate embedded in the plot,
  not a measured artifact.

Keep this snapshot unchanged. New benchmark claims must come from a complete
current-format run directory.
