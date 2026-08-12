# Chronological record of the noise-floor error

This file exists because the note's §3.2 claims that *the most precise available
estimate was the most misleading one*, and that claim is only evidenced by the
sequence in which the estimates were made. Reporting the final figure alone would
not support it.

**On the dates below.** Stage 5 carries ISO timestamps written by the measurement
scripts themselves; they are inside `data/determinism_control.jsonl` and
`data/reload_probe.jsonl` and can be checked by any reader. Stages 1-4 produced
artefacts that do not embed a timestamp, so their dates are reconstructed from file
modification times and the working log and are given to the day only as an
approximation.

What §3.2 of the note relies on is the **order** of the estimates - 2%, then 10.6%,
then a spread of exactly zero, then 22.5% - not the calendar. That order is fixed by
the artefacts: each stage's script consumes the previous stage's output or adds the
flag the previous stage's failure motivated. The dates are context, not evidence.

---

## Stage 1 — a 2% noise floor, from two measurements

Two measurements of `D(W+C)` were taken and their difference was adopted as the
noise floor: **≈ 2%**.

Every comparison in the sweep was adjudicated against this figure. Differences
larger than 2% were called real.

**What was wrong:** two measurements cannot estimate a spread. More importantly,
both were taken in a way that did not vary the factor that actually mattered.

Artefact: `data/e1_results.json` (reconstructed date 2026-07-29)

---

## Stage 2 — revised upward to 10.6%, and every earlier verdict re-adjudicated

Two further measurements widened the observed range:

```
mean 0.06085   spread 10.6%
```

The floor was revised from 2% to **10.6%**, this was recorded as a correction, and
**every comparison previously called significant was re-adjudicated. One of them
flipped.**

Conclusion recorded at the time:

> Measurement precision is a function of prompt length. The noise floor must be
> measured, not assumed.

`--wc-repeats` (default 3) was added so that the reference condition is repeated
and its observed spread used as the floor.

Artefacts: `data/e1_ttt_sweep.json` (reconstructed date 2026-07-30), `code/e1_perturbation_matched.py`

**What was still wrong:** the revision increased the *magnitude* of the floor but
did not change the *model of the variation*. Both estimates were of the wrong
quantity. Rigour applied to the wrong estimand converges confidently on the wrong
answer.

---

## Stage 3 — repeating the reference properly returned a spread of exactly zero

With `--wc-repeats` in place, the reference was repeated five times inside one
process:

```
0.05859649523664881
0.05859649523664881
0.05859649523664881
spread = 0.0
```

Five repetitions in a different process likewise returned `0.0521446` five times,
with no variation in any digit.

**Read naively this says the instrument is noiseless and every comparison is
licensed. It is the single most misleading number produced in this project.** A
within-process repetition estimates within-process variation; it cannot see
across-process variation by construction, and it reports the absence with maximal
confidence.

Artefact: `data/e1_perturbation_matched.json` (reconstructed date 2026-07-31)

---

## Stage 4 — the across-process spread: 22.5%

The same quantity, same code, same seeds, same probes, same machine, measured in
six separate processes:

```
0.0600316  0.0589731  0.0589650  0.0654272  0.0521446  0.0585965
(max − min) / mean = 22.5%
```

Two processes measuring an identical rank-256 adapter alongside their own
in-process reference:

| | D(rank 256) | D(W+C), same process | ratio |
|---|---|---|---|
| process B | 0.0623606 | 0.0654272 | 0.953 |
| process C | 0.0880676 | 0.0521446 | 1.689 |

Absolute values differ by 41%; **normalising by the in-process reference enlarged
the discrepancy to 77%**, because reference and treatment moved in opposite
directions.

---

## What was retracted

```
D ≈ 0.0107 · rank^0.318        high R², points visibly collinear in log-log
crossover ≈ rank 214–298       derived from the above
```

Both withdrawn. The fit had placed points from different instruments on one line.
Re-measuring all five ranks inside a single process put the crossover at
**rank 128, directly observed** (ratio 0.996), requiring no extrapolation.

A second, smaller lesson: even confined to one process, the aggregate fit
`D ≈ 0.00499 · rank^0.541` (R² = 0.981) extrapolates the crossover to rank 95
against the observed 128 — a 26% error. High R² in log-log coordinates is
compatible with local exponents varying by a factor of two.

---

## Stage 5 — the controls that could have overturned the diagnosis, and did not

**Determinism switches** (2026-08-02, timestamped in `data/determinism_control.jsonl`: six runs between 06:16 and 09:28 UTC): six fresh
processes, three with `torch.use_deterministic_algorithms(True)`,
`cudnn.deterministic = True`, `cudnn.benchmark = False`, TF32 disabled and
`CUBLAS_WORKSPACE_CONFIG=:4096:8` exported before interpreter start.

```
within-process spread   0.0000%  (all six)
across-process spread   15.86% off / 11.91% on
```

We do **not** claim the flags helped: a range statistic over three points is
extremely unstable. The supported claim is that the across-process variation is not
zero with the flags on.

**Reload probe** (2026-08-02, timestamped in `data/reload_probe.jsonl`: 13:04 and 14:15 UTC): four independent model
loads per process, two processes. Every measurement within a process was
bit-identical across all four loads; across processes the spread was 14.52%. The
differing state is therefore established at process or CUDA-context initialisation,
not at model load.

---

## The rule this produced

> Before trusting a noise floor, ask what the repetitions varied. If the answer is
> "nothing that distinguishes the cases I intend to compare", the floor licenses
> nothing, however small it is.
