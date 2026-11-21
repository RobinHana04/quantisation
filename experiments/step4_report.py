import re, os
N = 4022468096
f16_ppl = float(re.search(r"Final estimate: PPL = ([\d.]+)", open("logs/ppl-f16.log").read()).group(1))
def get(log):
    s = open(log).read(); g = lambda p: float(re.search(p, s).group(1))
    return g(r"Mean PPL\(Q\)\s*:\s*([\d.]+)"), g(r"Mean    KLD:\s*([\d.]+)"), g(r"Same top p:\s*([\d.]+)")
L = ["| type | size (GB) | bpw | PPL | ΔPPL | KLD without | KLD with imatrix | KLD reduction | top-1 without | top-1 with |", "|---|---|---|---|---|---|---|---|---|---|"]
for t in ["Q8_0", "Q6_K", "Q4_K_M", "Q3_K_M", "Q2_K", "IQ2_XS"]:
    im = get(f"logs/kl-{t}-im.log"); sz = os.path.getsize(f"gguf/qwen3-4b-{t}-im.gguf")
    if os.path.exists(f"logs/kl-{t}.log"):
        no = get(f"logs/kl-{t}.log"); kn, tn, red = f"{no[1]:.4f}", f"{no[2]:.1f}", f"{100*(1-im[1]/no[1]):.0f}%"
        pn = f"{no[0]:.2f} → "
    else: kn = tn = red = "n/a"; pn = ""
    L.append(f"| {t} | {sz/1e9:.2f} | {sz*8/N:.2f} | {pn}{im[0]:.2f} | {100*(im[0]/f16_ppl-1):+.1f}% | {kn} | {im[1]:.4f} | {red} | {tn} | {im[2]:.1f} |")
tab = "\n".join(L); print(tab)
txt = open("results.md").read().split("\n## Step 4")[0]
open("results.md", "w").write(txt + f"\n## Step 4: imatrix (calibration: WikiText-2 train, 200 chunks x 512; evaluation: test, 100 chunks)\n\n{tab}\n\nPPL: 'without → with'. ΔPPL vs F16 ({f16_ppl:.3f}). IQ2_XS does not exist without imatrix.\n")
