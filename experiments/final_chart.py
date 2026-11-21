import re, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
def get(log):
    s = open(log).read(); g = lambda p: float(re.search(p, s).group(1))
    return g(r"Mean PPL\(Q\)\s*:\s*([\d.]+)"), g(r"Mean    KLD:\s*([\d.]+)")
sz = lambda f: os.path.getsize(f) / 1e9
f16 = float(re.search(r"Final estimate: PPL = ([\d.]+)", open("logs/ppl-f16.log").read()).group(1))
T = ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"]
plain = [(t, sz(f"gguf/qwen3-4b-{t}.gguf"), *get(f"logs/kl-{t}.log")) for t in T]
im = [(t, sz(f"gguf/qwen3-4b-{t}-im.gguf"), *get(f"logs/kl-{t}-im.log")) for t in ["Q8_0", "Q6_K", "Q4_K_M", "Q3_K_M", "Q2_K", "IQ2_XS"]]
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for a, k, lab in ((ax[0], 2, "Perplexity (lower = better)"), (ax[1], 3, "KL divergence vs F16 (lower = better)")):
    a.plot([8.05] + [p[1] for p in plain], [f16 if k == 2 else 0] + [p[k] for p in plain], "o-", label="without imatrix")
    a.plot([p[1] for p in im], [p[k] for p in im], "s--", label="with imatrix")
    for p in plain: a.annotate(p[0], (p[1], p[k]), textcoords="offset points", xytext=(5, -10), fontsize=8)
    a.annotate("IQ2_XS", (im[-1][1], im[-1][k]), textcoords="offset points", xytext=(5, 5), fontsize=8)
    a.set_xlabel("file size (GB), Qwen3-4B"); a.set_ylabel(lab); a.grid(alpha=.3); a.legend()
ax[0].set_ylim(13.5, 25); ax[1].set_yscale("symlog", linthresh=1e-3)
ax[0].set_title("Size vs perplexity (WikiText-2 test; Q2_K without imatrix is off-chart: 39.2)", fontsize=8)
ax[1].set_title("Size vs KL divergence (symlog axis)", fontsize=8)
plt.tight_layout(); plt.savefig("result.png", dpi=130)
