# Real SWE-bench evaluation

This harness uses a literal upstream GitHub checkout, not the small `impl.py`
fixtures in `benchmark/tasks`.

The single canonical Mac-local case is `pytest-dev__pytest-11143`. It checks
out `pytest-dev/pytest@6995257cf470d2143ad1683824962de4071c0eb7`, gives only the
original issue to Claude Code, saves the model's patch, then applies the
official hidden test patch. It runs the official `FAIL_TO_PASS` test and the
entire affected test file as a stricter, macOS-compatible superset of the
`PASS_TO_PASS` selectors.

```bash
python benchmark/real_swebench.py --label sonnet-opus

python benchmark/real_swebench.py \
  --label opus-only \
  --strategy agentic-coding-fixed-opus
```

Results are written under `benchmark/results-real/<label>/`. The result JSON
records the exact repository commit, strategy, official test selectors, patch
hashes, agent exit status, and test status. Metrics remain in the same result
directory for cost comparison. The real repository checkout and Python 3.11
environment are prepared automatically with `uv` and run locally on macOS.
Each run receives a unique semantically inert cache namespace in its prompt so
one strategy cannot borrow a static-prefix cache created by the strategy run
immediately before it.

## Verified comparison (2026-07-23, cache-aware policy)

Across two counterbalanced isolated pairs, both strategies produced the same
official patch (SHA-256
`6574b635425adbf73a6238c2a15c17bd4076981843ee5c1d43782efe0ec89a8f`).
All four runs passed the issue-specific hidden test and the affected-file
regression suite (`115 passed, 1 skipped`). SmartAsk routed this obvious coding
instruction directly to Opus without an LLM classifier, kept Opus through the
coding loop, then used compact Sonnet for the final user-visible summary.

| Trial | Opus-only | SmartAsk | Saving | Quality |
|---|---:|---:|---:|---:|
| Isolated A | $0.893277 | $0.548733 | 38.6% | PASS / PASS |
| Isolated B, reversed order | $0.408359 | $0.471888 | -15.6% | PASS / PASS |
| **Aggregate** | **$1.301636** | **$1.020621** | **21.59%** | **2/2 / 2/2** |

SmartAsk saved `$0.281015` across the repeated real-repository evaluation. Its
two sessions used five and six Opus coding responses respectively, plus one
Sonnet finalizer each. Neither paid for a classifier. The Opus-only controls
used six and seven Opus responses.

The reversed trial demonstrates why one run is insufficient: provider cache
hits varied enough for Opus-only to win that pair. The aggregate is cheaper,
but SmartAsk does not promise that every stochastic session will beat every
Opus session. To reduce this variance in interactive use, the transport keeps
recent user/tool cache boundaries explicit, and the strategy leases a recently
used hard model for five minutes across human messages. A hot Opus cache read
can be cheaper than a cold Sonnet cache write; the lease avoids that isolated
switch while compact finalization remains cheap because it sends only bounded
evidence.

For deterministic messages such as `Reply with exactly ...; do not use tools`,
SmartAsk also skips the classifier and routes directly. Whether Sonnet or Opus
is cheaper for a single large-context message depends on which model currently
has a live provider cache; this is why the five-minute session lease is part of
the product policy rather than claiming a fixed easy-turn percentage.
