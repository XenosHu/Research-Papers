# Within-Process Determinism Is Not Reproducibility

Measurement code and raw outputs for the methods note *Within-Process Determinism
Is Not Reproducibility: A Measurement Hazard in Behavioural Comparison of Language
Models* (Xiangyu Hu, ORCID [0009-0008-7742-2726](https://orcid.org/0009-0008-7742-2726)).

## The observation in one paragraph

A behavioural divergence measure `D`, repeated inside a single process with the
model loaded once, returned **bit-identical values on every repetition — a spread
of exactly zero**. The same quantity, measured in separate processes with identical
seeds, prompts, code and hardware, varied by **22.5%**. An identical rank-256
adapter measured `0.0623606` in one process and `0.0880676` in another. Normalising
each measurement by a reference measured in the *same* process did not remove the
discrepancy but **enlarged it from 41% to 77%**, because the reference and the
treatment drifted in opposite directions.

Enabling PyTorch's deterministic-execution switches does not remove the effect.
Reloading the model inside a process does not perturb the result at all — four
independent loads gave bit-identical values — which locates the differing state at
**process initialisation** rather than at model load.

## Repository contents

```
code/
  e1_perturbation_matched.py   the original measurement, including --wc-repeats
  determinism_control.py       six processes, three with determinism switches on
  reload_probe.py              four model loads per process, two processes
data/
  e1_results.json              earliest run
  e1_ttt_sweep.json            rank/step sweep
  e1_perturbation_matched.json single-process re-measurement of all ranks
  determinism_control.jsonl    determinism on/off, one record per process
  reload_probe.jsonl           reload probe, one record per process
RECORD.md                      chronological record of two incorrect noise-floor
                               estimates and their correction
```

## Why `RECORD.md` is in this repository

The note reports two noise floors we estimated and then had to withdraw (2%, then
10.6%), and a power-law fit and derived crossover point that were retracted. That
sequence — 2%, then 10.6%, then a spread of exactly zero, then 22.5% — **is the
evidence** for the claim that the most precise available estimate can be the most
misleading one. A cleaned-up presentation reporting only the final figure would not
support the note's argument, so the record is published with the data.

## The diagnostic

> Measure the same quantity *n* times inside one process, and once in each of *n*
> processes. Report both spreads.

- Both spreads small → comparison across processes is licensed.
- Both comparable and non-zero → ordinary stochastic noise.
- **Within-process spread near zero, across-process spread large** → the regime
  described here. Values are not portable across processes, and neither are ratios
  against an in-process reference.

## Reproducing

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch transformers peft
python code/e1_perturbation_matched.py --wc-repeats 3
python code/determinism_control.py
python code/reload_probe.py
```

All runs use `Qwen/Qwen3-1.7B` in bfloat16 on a single CUDA device.

**Note that the phenomenon this repository documents means you should not expect to
reproduce our absolute numbers.** What should reproduce is the *shape*: a
within-process spread at or near zero, and a substantially larger across-process
spread on the long-prompt arm, with the short-prompt arm at exactly zero in every
process.

## Scope

One model, one size, one GPU, one framework version, one dtype, one metric; three
processes per determinism condition and two for the reload probe. We localise the
differing state in time (process initialisation) but do **not** identify the
mechanism, and we caution against reading "CUDA context" as an explanation rather
than a location.

## Citation

```bibtex
@misc{hu2026withinprocess,
  author = {Hu, Xiangyu},
  title  = {Within-Process Determinism Is Not Reproducibility: A Measurement
            Hazard in Behavioural Comparison of Language Models},
  year   = {2026},
  note   = {Preprint. arXiv identifier to be added on announcement;
            this entry is updated in place once assigned.}
}
```

## Licence

[CC BY 4.0](LICENSE).
