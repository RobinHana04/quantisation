"""Step 3: parse logs -> table in results.md + chart."""
import re, csv, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

N_PARAMS = 4022468096
types = ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"]
def bench(t):
    pp = tg = None
    for r in csv.DictReader(open(f"logs/bench-{t}.csv")):
        if r["n_prompt"] != "0": pp = float(r["avg_ts"])
        if r["n_gen"] != "0": tg = float(r["avg_ts"])
    size = int(r["model_size"])
    return size, pp, tg

rows = []
f16_size, f16_pp, f16_tg = bench("f16")
f16_ppl = float(re.search(r"Final estimate: PPL = ([\d.]+)", open("logs/ppl-f16.log").read()).group(1))  # real F16 run; PPL(base) from the KL log is lossy
rows.append(("F16", f16_size, f16_ppl, 0.0, 100.0, f16_pp, f16_tg))
for t in types:
    log = open(f"logs/kl-{t}.log").read()
    g = lambda p: float(re.search(p, log).group(1))
    size, pp, tg = bench(t)
    rows.append((t, size, g(r"Mean PPL\(Q\)\s*:\s*([\d.]+)"), g(r"Mean    KLD:\s*([\d.]+)"),
                 g(r"Same top p:\s*([\d.]+)"), pp, tg))
    base_in_kl = g(r"Mean PPL\(base\)\s*:\s*([\d.]+)")

hdr = "| variant | size (GB) | bpw | PPL | ΔPPL vs F16 | KLD | same top-1 % | pp t/s | tg t/s | tg-speedup |\n|---|---|---|---|---|---|---|---|---|---|"
lines = [hdr]
for n, s, ppl, kl, top, pp, tg in rows:
    lines.append(f"| {n} | {s/1e9:.2f} | {s*8/N_PARAMS:.2f} | {ppl:.3f} | {100*(ppl/f16_ppl-1):+.1f}% | {kl:.4f} | {top:.1f} | {pp:.0f} | {tg:.1f} | {tg/f16_tg:.2f}x |")
table = "\n".join(lines)
print(table); print("PPL(base) as seen in the KL run:", base_in_kl)
txt = open("results.md").read(); txt = txt.split("\n## Step 3")[0]; open("results.md", "w").write(txt)
open("results.md", "a").write(f"\n## Step 3: Qwen3-4B, GGUF variants (WikiText-2 test, ctx 512, 100 chunks)\n\n{table}\n\n![size vs ppl/kl](step3_size_vs_quality.png)\n")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for a, idx, lab in ((ax[0], 2, "Perplexity (lower = better)"), (ax[1], 3, "KL divergence vs F16 (lower = better)")):
    xs = [r[1] / 1e9 for r in rows]; ys = [r[idx] for r in rows]
    a.plot(xs, ys, "o-")
    for r in rows: a.annotate(r[0], (r[1] / 1e9, r[idx]), textcoords="offset points", xytext=(4, 5), fontsize=8)
    a.set_xlabel("file size (GB)"); a.set_ylabel(lab); a.grid(alpha=.3)
ax[1].set_yscale("symlog", linthresh=1e-3)
plt.tight_layout(); plt.savefig("step3_size_vs_quality.png", dpi=130)
