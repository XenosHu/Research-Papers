"""
determinism_control.py -- does enabling deterministic algorithms remove the
cross-process variation reported in the methods note?

WHY THIS SCRIPT EXISTS
----------------------
The note "Within-Process Determinism Is Not Reproducibility" reports that D(W+C)
is bit-exact within a process and varies by 22.5% across processes. Section 5.3
of that note states plainly that we never tested whether the standard PyTorch
determinism switches eliminate the effect. This script is that test. It is the
first question a referee will ask and we should not let them ask it first.

The answer changes what the note claims:

  * If the flags REMOVE the effect, the recommendation collapses to "set the
    flags", and the note's contribution shrinks to a quantification of the cost
    of not setting them. Still publishable, much smaller.
  * If the flags DO NOT remove it, the hazard is not a configuration mistake and
    the note stands as written.
  * If the flags remove it at a large speed cost, both facts go in the paper.

There is no outcome of this script that is bad for the project. There is only an
outcome that is bad for a claim, and we would rather find it here.

WHAT IT MEASURES
----------------
Two arms from e1_boundary_ordering, inference only, no LoRA training (so this is
minutes, not hours):

  W    frozen model, no context   -- SHORT prompts. Reproduced exactly in every
                                     process so far. Acts as the built-in control
                                     of note section 5.2.
  W+C  twenty turns in context    -- LONG prompts. This is the arm that drifted.

Each process repeats the W+C measurement --repeats times WITHOUT reloading, which
gives the within-process spread, then writes one record to a JSONL file. Running
the script in several separate processes gives the across-process spread. The
whole point is that these two numbers are different quantities, so the script
refuses to report either one alone.

DETERMINISM SETTINGS APPLIED WHEN --determinism on
--------------------------------------------------
  CUBLAS_WORKSPACE_CONFIG=:4096:8     (env, must precede CUDA context creation)
  torch.use_deterministic_algorithms(True)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  torch.backends.cuda.matmul.allow_tf32 = False
  torch.backends.cudnn.allow_tf32 = False

The env var is set by the DRIVER in the child's environment, before the child
interpreter starts. Setting it from inside a running process that has already
initialised cuBLAS is too late and silently does nothing -- a failure mode worth
knowing about, and the reason `run` exists rather than a loop inside one process.

USAGE
-----
    # do everything: 3 processes off, 3 processes on, then report
    python determinism_control.py run --runs 3

    # one process by hand (what `run` spawns)
    python determinism_control.py measure --determinism on

    # re-print the verdict from an existing log
    python determinism_control.py analyze

Runtime is roughly (probes x n x 4 conditions) forward passes per arm, times
(1 + repeats) for W+C. With defaults, a few minutes per process on a 1.7B model.

Author: Claude Opus 5, 2026-08-02. Not yet run by its author -- no GPU in the
authoring sandbox. Numbers come from Xenos's machine or they do not exist.
"""
import argparse, json, os, subprocess, sys, time, platform
from datetime import datetime, timezone

LOG = "determinism_control.jsonl"

DET_ENV = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "TOKENIZERS_PARALLELISM": "false",
}


# --------------------------------------------------------------------- measure
def apply_determinism(torch, on):
    """Returns (applied_ok, message). Called AFTER torch import, BEFORE model load."""
    if not on:
        torch.backends.cudnn.benchmark = True
        return True, "determinism off (cudnn.benchmark=True, tf32 left at default)"
    msgs = []
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != DET_ENV["CUBLAS_WORKSPACE_CONFIG"]:
        msgs.append("WARNING: CUBLAS_WORKSPACE_CONFIG not set in the environment. "
                    "Setting it now is TOO LATE if cuBLAS is already initialised. "
                    "Use `run`, or export it before launching python.")
    try:
        torch.use_deterministic_algorithms(True)
        msgs.append("use_deterministic_algorithms(True) ok")
    except Exception as e:                       # some ops have no deterministic impl
        return False, f"use_deterministic_algorithms failed: {e}"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        msgs.append("tf32 disabled")
    except Exception as e:
        msgs.append(f"tf32 toggle unavailable: {e}")
    return True, "; ".join(msgs)


def measure(args):
    import numpy as np
    import torch
    import e1_boundary_ordering as E1

    ok, det_msg = apply_determinism(torch, args.determinism == "on")
    print(f"[determinism={args.determinism}] {det_msg}")
    if not ok and args.determinism == "on":
        rec = dict(error="determinism_not_applicable", detail=det_msg)
        _append(rec, args)
        print("RECORDED AS FAILURE. This is a real result: it means the model's own "
              "kernels have no deterministic implementation, and 'just set the flags' "
              "is not available advice.")
        return

    import random
    np.random.seed(E1.SEED); random.seed(E1.SEED)
    probes = E1.build_probes(args.probes)

    t_load = time.time()
    runner = E1.Runner(args.model)
    load_s = time.time() - t_load

    def arm(fn):
        by_cond = {}
        for cond in E1.CONDITIONS:
            hists = E1.build_histories(cond, args.n)
            dists, _ = fn(runner, hists, probes)
            by_cond[cond] = dists
        w, b = E1.divergences(by_cond)
        return b - w

    t0 = time.time()
    D_W = arm(E1.arm_W)
    t_W = time.time() - t0

    wc = []
    t0 = time.time()
    for i in range(args.repeats):
        wc.append(arm(E1.arm_WC))
        print(f"  W+C repeat {i+1}/{args.repeats}: {wc[-1]:.17g}")
    t_WC = (time.time() - t0) / max(args.repeats, 1)

    spread = (max(wc) - min(wc)) / abs(np.mean(wc)) if np.mean(wc) else 0.0
    rec = dict(
        ts=datetime.now(timezone.utc).isoformat(),
        pid=os.getpid(),
        determinism=args.determinism,
        det_msg=det_msg,
        model=args.model,
        probes=args.probes, n=args.n, repeats=args.repeats,
        D_W=float(D_W),
        D_WC_values=[float(x) for x in wc],
        D_WC_mean=float(np.mean(wc)),
        D_WC_within_spread=float(spread),
        seconds_load=load_s, seconds_W=t_W, seconds_WC_per_repeat=t_WC,
        torch=torch.__version__,
        cuda=getattr(torch.version, "cuda", None),
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        cublas_ws=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        python=platform.python_version(),
    )
    _append(rec, args)
    print(f"\nD(W)   = {D_W:.17g}   <- expected exactly 0.0 in every process")
    print(f"D(W+C) = {rec['D_WC_mean']:.17g}   within-process spread = {spread:.4%}")
    print(f"written: {args.log}")


def _append(rec, args):
    with open(args.log, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


# --------------------------------------------------------------------- driver
def run(args):
    """Spawn separate processes. Separate processes are the whole experiment --
    a loop inside one process cannot produce the variation being tested for."""
    here = os.path.dirname(os.path.abspath(__file__))
    for setting in ["off", "on"]:
        for i in range(args.runs):
            env = dict(os.environ)
            env.update(DET_ENV if setting == "on" else
                       {"TOKENIZERS_PARALLELISM": "false"})
            if setting == "off":
                env.pop("CUBLAS_WORKSPACE_CONFIG", None)
            cmd = [sys.executable, os.path.abspath(__file__), "measure",
                   "--determinism", setting, "--model", args.model,
                   "--probes", str(args.probes), "--n", str(args.n),
                   "--repeats", str(args.repeats), "--log", args.log]
            print(f"\n=== process {i+1}/{args.runs}, determinism={setting} ===")
            subprocess.run(cmd, env=env, cwd=here, check=False)
    analyze(args)


# --------------------------------------------------------------------- analyze
def analyze(args):
    import numpy as np
    if not os.path.exists(args.log):
        print(f"no log at {args.log}"); return
    recs = [json.loads(l) for l in open(args.log, encoding="utf-8") if l.strip()]
    recs = [r for r in recs if "error" not in r]
    errs = [json.loads(l) for l in open(args.log, encoding="utf-8")
            if l.strip() and "error" in json.loads(l)]

    print("\n" + "=" * 72)
    print("DETERMINISM CONTROL")
    print("=" * 72)
    if errs:
        print(f"\n{len(errs)} process(es) could not apply determinism at all:")
        for e in errs:
            print("  " + e.get("detail", "")[:200])

    summary = {}
    for setting in ["off", "on"]:
        rs = [r for r in recs if r["determinism"] == setting]
        if not rs:
            continue
        means = [r["D_WC_mean"] for r in rs]
        within = [r["D_WC_within_spread"] for r in rs]
        across = (max(means) - min(means)) / abs(np.mean(means)) if len(means) > 1 else float("nan")
        w_exact = all(r["D_W"] == 0.0 for r in rs)
        secs = float(np.mean([r["seconds_WC_per_repeat"] for r in rs]))
        summary[setting] = dict(n=len(rs), across=across, within_max=max(within),
                                secs=secs, w_exact=w_exact, means=means)

        print(f"\n--- determinism {setting} --- ({len(rs)} processes)")
        for r in rs:
            print(f"  pid {r['pid']:>7}  D(W+C) = {r['D_WC_mean']:.17g}   "
                  f"within-process spread {r['D_WC_within_spread']:.4%}   "
                  f"D(W) = {r['D_W']:.3g}")
        print(f"  WITHIN-process spread (worst): {max(within):.4%}")
        print(f"  ACROSS-process spread        : {across:.4%}")
        print(f"  D(W) exactly 0 in every process: {'YES' if w_exact else 'NO'}")
        print(f"  seconds per W+C measurement  : {secs:.1f}")

    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    if "on" not in summary or "off" not in summary:
        print("Need at least one process in each setting. Run `run --runs 3`.")
        return
    if summary["off"]["n"] < 2 or summary["on"]["n"] < 2:
        print("Need at least TWO processes per setting -- across-process spread is "
              "undefined with one. Re-run with --runs 3.")
        return

    off_a, on_a = summary["off"]["across"], summary["on"]["across"]
    slow = summary["on"]["secs"] / summary["off"]["secs"] if summary["off"]["secs"] else float("nan")

    print(f"across-process spread   off: {off_a:.4%}    on: {on_a:.4%}")
    print(f"deterministic-mode slowdown: {slow:.2f}x")
    print()

    if on_a < 0.005 and off_a > 0.02:
        print("*** FLAGS REMOVE THE EFFECT. ***")
        print("Section 5.3 of the note must be rewritten. The recommendation becomes")
        print("'enable deterministic algorithms', and the note's contribution narrows")
        print("to quantifying the cost of not doing so. Report the slowdown above as")
        print("the price, since that is why the flags are not the default.")
    elif on_a > 0.02:
        print("*** FLAGS DO NOT REMOVE THE EFFECT. ***")
        print("The hazard is not a configuration mistake. Section 5.3 stands, and this")
        print("run becomes a positive result in the paper rather than an admitted gap.")
        print("Next question: is the residual variation in the model load (weight")
        print("layout, allocator) rather than in the kernels?")
    else:
        print("*** PARTIAL. ***")
        print("Reduced but not eliminated. Report both numbers; do not round the")
        print("residual away. A recommendation of 'set the flags' would be incomplete.")

    if not summary["off"]["w_exact"] or not summary["on"]["w_exact"]:
        print("\n!! D(W) was NOT exactly 0 in some process. The short-prompt control")
        print("   from section 5.2 has failed, which means the phenomenon is not")
        print("   confined to long inputs and the note's diagnosis needs revision.")
        print("   This overrides the verdict above -- investigate before writing.")

    print("\nPaste the block above into METHODS_NOTE section 5.3 verbatim. Whatever it")
    print("says. The value of the note comes from it having no untested claims in it.")


# --------------------------------------------------------------------- cli
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--model", default="Qwen/Qwen3-1.7B")
        p.add_argument("--probes", type=int, default=200)
        p.add_argument("--n", type=int, default=2)
        p.add_argument("--repeats", type=int, default=3)
        p.add_argument("--log", default=LOG)

    m = sub.add_parser("measure", help="one process, one setting")
    common(m); m.add_argument("--determinism", choices=["on", "off"], required=True)

    r = sub.add_parser("run", help="spawn N processes in each setting, then analyze")
    common(r); r.add_argument("--runs", type=int, default=3)

    a = sub.add_parser("analyze", help="report from an existing log")
    common(a)

    args = ap.parse_args()
    {"measure": measure, "run": run, "analyze": analyze}[args.cmd](args)


if __name__ == "__main__":
    main()
