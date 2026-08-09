"""
reload_probe.py -- WHERE is the differing state established?

THE QUESTION
------------
determinism_control.py settled that the across-process variation in D(W+C) is
real and survives torch.use_deterministic_algorithms(True):

    within-process spread   0.0000%   (18/18 measurements bit-identical)
    across-process spread  15.9% off / 11.9% on

So something is fixed once per process and differs between processes. This
script asks WHERE it is fixed. There are two candidates and they are separated
by one cheap manipulation: reloading the model inside a single process.

    L1  measure, then measure again, same loaded model      -> known: spread 0
    L2  measure, RELOAD the model, measure again, ...       -> this script
    L3  measure in a fresh process                          -> known: 12-16%

  If L2 ~ L3, the state is established AT MODEL LOAD -- weight memory layout,
      allocator placement, or a per-shape autotuning cache populated the first
      time each tensor shape is seen. Practical consequence: reloading a model
      inside a long-running service silently re-rolls the measurement, so a
      persistent server is no safer than separate processes.

  If L2 ~ L1 (zero), the state belongs to the CUDA CONTEXT and survives reload.
      Practical consequence: one process is genuinely one instrument, and
      "complete every comparison in one process" is sufficient advice.

Both answers are publishable and they give different advice, which is what makes
this worth 30 minutes. Note that cuDNN's benchmark cache is per-process and is
NOT cleared by deleting the model, so the two hypotheses really do come apart
here rather than being two names for one thing.

WHAT IT DOES NOT SETTLE
-----------------------
It localises the state in time (before vs after model load). It does not
identify the mechanism. Do not write "caused by allocator placement" on the
strength of this script; write "established at model load".

USAGE
-----
    python reload_probe.py run --runs 2          # 2 processes x 4 loads
    python reload_probe.py measure               # one process by hand
    python reload_probe.py analyze

Roughly 9-10 min per measurement at the defaults, so ~40 min per process.
Use --loads 3 --probes 120 to halve it if you want a first look sooner; the
comparison stays internally valid because every number in it is produced the
same way. Do NOT compare its absolute values against the note's other runs.

Author: Claude Opus 5, 2026-08-02. Not run by its author -- no GPU in the
authoring sandbox.
"""
import argparse, gc, json, os, subprocess, sys, platform, time
from datetime import datetime, timezone

LOG = "reload_probe.jsonl"


def measure(args):
    import numpy as np
    import torch
    import e1_boundary_ordering as E1
    import random

    if args.determinism == "on":
        try:
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception as e:
            print(f"could not enable determinism: {e}")
    else:
        torch.backends.cudnn.benchmark = True

    np.random.seed(E1.SEED); random.seed(E1.SEED)
    probes = E1.build_probes(args.probes)

    def measure_WC(runner):
        by_cond = {}
        for cond in E1.CONDITIONS:
            hists = E1.build_histories(cond, args.n)
            dists, _ = E1.arm_WC(runner, hists, probes)
            by_cond[cond] = dists
        w, b = E1.divergences(by_cond)
        return b - w

    def measure_W(runner):
        by_cond = {}
        for cond in E1.CONDITIONS:
            hists = E1.build_histories(cond, args.n)
            dists, _ = E1.arm_W(runner, hists, probes)
            by_cond[cond] = dists
        w, b = E1.divergences(by_cond)
        return b - w

    loads = []
    for i in range(args.loads):
        t0 = time.time()
        runner = E1.Runner(args.model)
        load_s = time.time() - t0

        # Two repeats on the FIRST load only. Their agreement is the internal
        # control: it re-establishes that the instrument is exact when nothing
        # is reloaded, in this process, before we start reloading. Without it a
        # non-zero L2 spread could be dismissed as ordinary noise.
        reps = 2 if i == 0 else 1
        vals = [measure_WC(runner) for _ in range(reps)]
        d_w = measure_W(runner) if i == 0 else None

        loads.append(dict(load_index=i, values=[float(v) for v in vals],
                          D_W=(float(d_w) if d_w is not None else None),
                          seconds_load=load_s))
        print(f"  load {i}: " + ", ".join(f"{v:.17g}" for v in vals))

        del runner
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    firsts = [l["values"][0] for l in loads]
    across_loads = (max(firsts) - min(firsts)) / abs(np.mean(firsts)) if len(firsts) > 1 else float("nan")
    same_load = loads[0]["values"]
    same_load_spread = (max(same_load) - min(same_load)) / abs(np.mean(same_load)) if len(same_load) > 1 else 0.0

    rec = dict(
        ts=datetime.now(timezone.utc).isoformat(), pid=os.getpid(),
        determinism=args.determinism, model=args.model,
        probes=args.probes, n=args.n, loads=args.loads,
        per_load=loads,
        first_of_each_load=[float(x) for x in firsts],
        across_load_spread=float(across_loads),
        same_load_spread=float(same_load_spread),
        torch=torch.__version__, cuda=getattr(torch.version, "cuda", None),
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        python=platform.python_version(),
    )
    with open(args.log, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")

    print(f"\n  same-load spread (control) : {same_load_spread:.4%}   <- expect 0")
    print(f"  ACROSS-RELOAD spread       : {across_loads:.4%}   <- the question")
    print(f"written: {args.log}")


def run(args):
    here = os.path.dirname(os.path.abspath(__file__))
    for i in range(args.runs):
        print(f"\n=== process {i+1}/{args.runs} ===")
        subprocess.run([sys.executable, os.path.abspath(__file__), "measure",
                        "--model", args.model, "--probes", str(args.probes),
                        "--n", str(args.n), "--loads", str(args.loads),
                        "--determinism", args.determinism, "--log", args.log],
                       cwd=here, check=False)
    analyze(args)


def analyze(args):
    import numpy as np
    if not os.path.exists(args.log):
        print(f"no log at {args.log}"); return
    recs = [json.loads(l) for l in open(args.log, encoding="utf-8") if l.strip()]
    if not recs:
        print("empty log"); return

    print("\n" + "=" * 72)
    print("RELOAD PROBE -- where is the differing state established?")
    print("=" * 72)

    for r in recs:
        print(f"\npid {r['pid']}  determinism={r['determinism']}  loads={r['loads']}")
        for l in r["per_load"]:
            vals = ", ".join(f"{v:.17g}" for v in l["values"])
            print(f"  load {l['load_index']}: {vals}")
        print(f"  same-load spread (control): {r['same_load_spread']:.4%}")
        print(f"  across-reload spread      : {r['across_load_spread']:.4%}")

    ctl = max(r["same_load_spread"] for r in recs)
    reload_spreads = [r["across_load_spread"] for r in recs if r["loads"] > 1]
    if not reload_spreads:
        print("\nneed --loads >= 2"); return
    worst = max(reload_spreads)

    # across-PROCESS spread, if more than one process is in this log
    firsts = [r["per_load"][0]["values"][0] for r in recs]
    across_proc = ((max(firsts) - min(firsts)) / abs(np.mean(firsts))
                   if len(firsts) > 1 else float("nan"))

    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    print(f"same-load spread (control)    : {ctl:.4%}")
    print(f"across-RELOAD spread (worst)  : {worst:.4%}")
    print(f"across-PROCESS spread         : {across_proc:.4%}   ({len(firsts)} processes)")
    print()

    if ctl > 0.001:
        print("!! CONTROL FAILED. The same loaded model gave different answers on")
        print("   two consecutive measurements. That contradicts 18/18 bit-identical")
        print("   repeats in determinism_control. Something changed in the setup --")
        print("   find it before interpreting anything below.")
        return

    if worst < 0.005:
        print("*** STATE SURVIVES RELOAD -> it belongs to the CUDA CONTEXT. ***")
        print("Reloading the model inside a process does not re-roll the measurement.")
        print("One process is genuinely one instrument, and the note's recommendation")
        print("('complete every comparison in one process') is sufficient as written.")
        print("Section 5.4 gets a definite answer instead of a proposed experiment.")
    elif worst > 0.03:
        print("*** STATE IS ESTABLISHED AT MODEL LOAD. ***")
        print("Reloading re-rolls the measurement, at a magnitude comparable to")
        print("starting a new process. This STRENGTHENS the note and adds a warning")
        print("it does not currently contain: a long-running service that reloads or")
        print("swaps models is no safer than separate processes, so 'one process' is")
        print("NOT sufficient -- the rule must be 'one load'.")
    else:
        print("*** INTERMEDIATE. ***")
        print("Reload perturbs the measurement but by less than a fresh process does.")
        print("Report both numbers and claim only that model load is ONE contributor,")
        print("not the whole of it. Do not round this into either clean story.")

    print("\nWhatever this says, it goes into section 5.4 as written. The note's")
    print("value is that it contains no untested claims.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--model", default="Qwen/Qwen3-1.7B")
        p.add_argument("--probes", type=int, default=200)
        p.add_argument("--n", type=int, default=2)
        p.add_argument("--loads", type=int, default=4)
        p.add_argument("--determinism", choices=["on", "off"], default="off")
        p.add_argument("--log", default=LOG)

    m = sub.add_parser("measure"); common(m)
    r = sub.add_parser("run"); common(r); r.add_argument("--runs", type=int, default=2)
    a = sub.add_parser("analyze"); common(a)

    args = ap.parse_args()
    {"measure": measure, "run": run, "analyze": analyze}[args.cmd](args)


if __name__ == "__main__":
    main()
