"""
e1_perturbation_matched.py -- compare where a weight update is written, at matched
perturbation size rather than at matched rank.

WHY THIS SCRIPT EXISTS. The previous run compared attention-targeted and MLP-targeted
LoRA at rank 8 and found the MLP arm produced LESS behavioural divergence
(0.0124 vs 0.0209), refuting the pre-registered prediction P-A. That refutation is not
interpretable, because the comparison was confounded:

    rank 8 on q_proj (2048 x 2048) and rank 8 on gate_proj (2048 x 6144) are not the
    same size perturbation. Equal rank is not equal capacity, and it is not equal
    relative change to the weight matrix. I matched the wrong quantity -- the fifth
    time on this project that I set a gate on something other than what carries the
    conclusion.

The fix is not to hand-tune ranks until they look comparable. It is to stop matching a
proxy and measure the thing itself. For every adapted module we compute

    delta = || dW ||_F / || W ||_F        where  dW = (alpha / r) * B @ A

which is dimension-invariant: it asks what fraction of the existing weight's magnitude
the update represents, regardless of the matrix's shape. We then report D as a function
of the MEASURED delta rather than as a function of rank or steps, so that the attention
and MLP curves can be compared where their deltas overlap.

Noise floor: D(W+C) came out as 0.0600316 and 0.0589731 on two runs with identical seeds
and probes, a spread of 1.8%. Differences below roughly 2% of D(W+C) are therefore not
meaningful. The script prints this reminder with the results.

Usage
    python e1_perturbation_matched.py --model Qwen/Qwen3-1.7B
    python e1_perturbation_matched.py --model Qwen/Qwen3-1.7B --ranks 8,64 --steps 200

Runtime is dominated by steps x ranks x module-sets x histories.
"""
import argparse, json, time
import numpy as np

import e1_boundary_ordering as E1
from e1_ttt_sweep import measure_arm, unload, MODULE_SETS

NOISE_FLOOR_FRAC = 0.02          # superseded -- see the note below, kept for the record

# NOISE FLOOR, CORRECTED 2026-07-31.
# I set the floor at 2% from two measurements of D(W+C). Four measurements now exist:
#   0.0600316, 0.0589731, 0.0589650, 0.0654272   -> spread 10.6%
# The 2% figure was wrong and every gap I called "above the noise floor" using it has
# to be rechecked. Note the asymmetry: D(W) is exactly 0 on every run, so the model is
# perfectly deterministic on the SHORT probe prompts. It is the LONG prompts of the W+C
# arm that vary, which is consistent with kernel/accumulation-order differences in
# bfloat16 at longer sequence lengths. Measurement precision is therefore a function of
# prompt length, and the floor must be measured, not assumed. --wc-repeats does that.


def relative_perturbation(peft_model):
    """Mean over adapted modules of ||dW||_F / ||W||_F, plus the per-module spread.

    dW is reconstructed from the adapter itself rather than by differencing weights,
    so this does not depend on the base model being reloaded."""
    import torch
    ratios = []
    for name, mod in peft_model.named_modules():
        A = getattr(mod, "lora_A", None)
        B = getattr(mod, "lora_B", None)
        if A is None or B is None:
            continue
        try:
            key = next(iter(A.keys()))          # usually "default"
            Aw = A[key].weight.detach().float()
            Bw = B[key].weight.detach().float()
        except Exception:
            continue
        base = getattr(mod, "base_layer", mod)
        W = getattr(base, "weight", None)
        if W is None:
            continue
        scaling = getattr(mod, "scaling", {})
        s = scaling.get(key, 1.0) if isinstance(scaling, dict) else float(scaling)
        with torch.no_grad():
            dW = s * (Bw @ Aw)
            r = (torch.linalg.norm(dW) / (torch.linalg.norm(W.detach().float()) + 1e-12))
        ratios.append(float(r))
    if not ratios:
        return None, None, 0
    return float(np.mean(ratios)), float(np.std(ratios)), len(ratios)


def lora_arm_measured(steps, lr, rank, modules, box):
    """Same as the sweep's LoRA arm, but records the induced perturbation per history."""
    def fn(runner, histories, probes):
        dists = []
        for hist in histories:
            m = runner.lora_train(hist, steps=steps, lr=lr, rank=rank,
                                  targets=MODULE_SETS[modules])
            mean, sd, n = relative_perturbation(m)
            if mean is not None:
                box.append(mean)
            dists.append([runner.next_token_dist(p, model=m)[0] for p in probes])
            unload(m)
        return dists, None
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--ranks", default="8,64")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--modules", default="attn,mlp")
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--probes", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wc-repeats", type=int, default=3,
                    help="how many times to measure the W+C reference. Its spread IS "
                         "the noise floor; do not assume one.")
    ap.add_argument("--out", default="e1_perturbation_matched.json")
    a = ap.parse_args()

    ranks = [int(r) for r in a.ranks.split(",")]
    module_sets = a.modules.split(",")
    print(f"model={a.model}  steps={a.steps}  ranks={ranks}  modules={module_sets}")
    print(f"histories/cond={a.n}  probes={a.probes}\n")

    np.random.seed(E1.SEED)
    probes = E1.build_probes(a.probes)
    runner = E1.Runner(a.model)

    print("[1/3] W reference")
    w_w, w_b, _ = measure_arm(E1.arm_W, runner, probes, a.n, "W")
    print(f"      D(W) = {w_b - w_w:.6g}   (must be ~0)\n")

    print(f"[2/3] W+C reference x{a.wc_repeats}  (its spread IS the noise floor)")
    wc = []
    for k in range(a.wc_repeats):
        c_w, c_b, _ = measure_arm(E1.arm_WC, runner, probes, a.n, f"W+C[{k+1}]")
        wc.append(c_b - c_w)
        print(f"      D(W+C)[{k+1}] = {wc[-1]:.6g}")
    d_wc = float(np.mean(wc))
    spread = float(max(wc) - min(wc)) if len(wc) > 1 else NOISE_FLOOR_FRAC * d_wc
    floor = max(spread, 1e-12)
    print(f"      mean D(W+C) = {d_wc:.6g}   spread = {spread:.6g} "
          f"({100*spread/d_wc:.1f}% of the mean)")
    print(f"      MEASURED noise floor = {floor:.6g} -- smaller gaps mean nothing\n")
    out_wc = dict(values=wc, mean=d_wc, spread=spread)

    print("[3/3] perturbation-matched comparison")
    rows = []
    for mods in module_sets:
        for r in ranks:
            box = []
            t0 = time.time()
            tw, tb, _ = measure_arm(
                lora_arm_measured(a.steps, a.lr, r, mods, box),
                runner, probes, a.n, f"{mods}/r{r}")
            d = tb - tw
            delta = float(np.mean(box)) if box else float("nan")
            rows.append(dict(modules=mods, rank=r, steps=a.steps, D=d,
                             delta=delta, ratio_to_WC=d / d_wc,
                             seconds=time.time() - t0))
            print(f"      {mods:>4}/r{r:<4} delta={delta:.5g}  D={d:.6g}"
                  f"  D/D(W+C)={d/d_wc:.3f}  [{time.time()-t0:.0f}s]\n", flush=True)

    print("=" * 72)
    print(f"{'modules':>8} {'rank':>5} {'delta (dW/W)':>14} {'D':>11} {'D/D(W+C)':>10}")
    for r in sorted(rows, key=lambda x: x["delta"]):
        print(f"{r['modules']:>8} {r['rank']:>5} {r['delta']:>14.5g}"
              f" {r['D']:>11.6g} {r['ratio_to_WC']:>10.3f}")

    # --- the actual comparison: at overlapping delta, does location matter? ---
    by_mod = {m: sorted([r for r in rows if r["modules"] == m],
                        key=lambda x: x["delta"]) for m in module_sets}
    print("\nlocation effect at comparable perturbation:")
    if len(module_sets) == 2 and all(by_mod[m] for m in module_sets):
        m1, m2 = module_sets
        lo = max(by_mod[m1][0]["delta"], by_mod[m2][0]["delta"])
        hi = min(by_mod[m1][-1]["delta"], by_mod[m2][-1]["delta"])
        if lo > hi:
            print("  NO OVERLAP in delta between the two module sets. The comparison")
            print("  cannot be made from this run. Widen --ranks so the ranges overlap;")
            print("  reporting a difference across non-overlapping perturbations would")
            print("  repeat exactly the confound this script exists to remove.")
        else:
            def interp(rs, x):
                xs = [r["delta"] for r in rs]; ys = [r["D"] for r in rs]
                return float(np.interp(x, xs, ys))
            mid = 0.5 * (lo + hi)
            d1, d2 = interp(by_mod[m1], mid), interp(by_mod[m2], mid)
            gap = abs(d1 - d2)
            print(f"  at delta = {mid:.5g}:  D({m1}) = {d1:.6g}   D({m2}) = {d2:.6g}")
            if gap < floor:
                print(f"  gap {gap:.6g} is BELOW the noise floor {floor:.6g}.")
                print("  Location does not detectably matter at this perturbation size.")
            else:
                bigger = m1 if d1 > d2 else m2
                print(f"  gap {gap:.6g} exceeds the noise floor. {bigger} produces more")
                print("  divergence at equal perturbation, so WHERE the update is written")
                print("  matters independently of HOW LARGE it is.")

    json.dump(dict(D_W=w_b - w_w, D_WC=d_wc, WC_repeats=out_wc,
                   noise_floor=floor, noise_floor_measured=True, rows=rows),
              open(a.out, "w"), indent=1)
    print(f"\nwritten: {a.out}")


if __name__ == "__main__":
    main()
