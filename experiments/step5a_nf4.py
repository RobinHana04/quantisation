"""Step 5A: NF4 (non-uniform levels) vs uniform Q4_0 style, on real weights. All from scratch."""
import numpy as np, glob
from safetensors import safe_open

NF4 = np.array([-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635,
    -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725,
    0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039108276,
    0.7229568362236023, 1.0], np.float32)

def f16(x): return x.astype(np.float16).astype(np.float32)

def nf4(B, double_quant=False):
    s = np.abs(B).max(1, keepdims=True)                      # absmax -> normalize to [-1,1]
    if double_quant:                                         # quantize the scales themselves to 8 bit, per 256 scales
        flat = s.ravel(); pad = (-len(flat)) % 256
        g = np.pad(flat, (0, pad)).reshape(-1, 256); mu = g.mean(1, keepdims=True); c = g - mu
        sc = np.abs(c).max(1, keepdims=True) / 127; sc[sc == 0] = 1
        s = ((np.round(c / sc) * sc + mu).ravel()[:len(flat)]).reshape(-1, 1)
    else:
        s = f16(s)
    s = np.where(s == 0, 1, s)
    n = B / s
    idx = np.abs(n[..., None] - NF4).argmin(-1)              # nearest level
    return s * NF4[idx]

def q4_0(B):                                                 # uniform, see step 2
    mx = B[np.arange(len(B)), np.abs(B).argmax(1)][:, None]
    d = f16(mx / -8); id_ = np.where(d != 0, 1 / np.where(d == 0, 1, d), 0)
    q = np.minimum(15, np.floor(B * id_ + 8.5))
    return d * (q - 8)

mse = lambda a, b: float(((a - b) ** 2).mean())
def kurt(x): z = (x - x.mean()) / x.std(); return float((z ** 4).mean())   # normal = 3

names = [f"model.layers.{l}.{n}" for l, n in [(10, "self_attn.q_proj"), (10, "self_attn.k_proj"), (10, "self_attn.v_proj"),
         (10, "self_attn.o_proj"), (10, "mlp.gate_proj"), (10, "mlp.up_proj"), (10, "mlp.down_proj"),
         (0, "mlp.gate_proj"), (27, "mlp.gate_proj")]]
W = {}
for f in glob.glob("models/Qwen3-1.7B/*.safetensors"):
    with safe_open(f, "pt") as g:
        for n in names:
            if n + ".weight" in g.keys(): W[n] = g.get_tensor(n + ".weight").float().numpy()

rows = ["| tensor | kurtosis | Q4_0 block32 (4.50 bpw) | NF4 block32 (4.50) | NF4/Q4_0 | Q4_0 block64 (4.25) | NF4 block64 (4.25) | NF4/Q4_0 | NF4 block64 + DQ (4.13) |",
        "|---|---|---|---|---|---|---|---|---|"]
tot = {k: [] for k in "abcde"}
for n in names:
    w = W[n]; r = []
    for bs in (32, 64):
        B = w.reshape(-1, bs); r += [mse(B, q4_0(B)), mse(B, nf4(B))]
    B = w.reshape(-1, 64); dq = mse(B, nf4(B, True))
    rows.append(f"| {n.replace('model.layers.','L')} | {kurt(w.ravel()):.1f} | {r[0]:.3e} | {r[1]:.3e} | {r[1]/r[0]:.2f} | {r[2]:.3e} | {r[3]:.3e} | {r[3]/r[2]:.2f} | {dq:.3e} |")
    tot["a"].append(r[1] / r[0]); tot["b"].append(r[3] / r[2])
tab = "\n".join(rows); print(tab)
print(f"\nmean NF4/Q4_0 MSE ratio: block32 {np.mean(tot['a']):.2f}, block64 {np.mean(tot['b']):.2f}")
open("results.md", "a").write(f"\n## Step 5A: NF4 vs Q4_0 (MSE on weights, Qwen3-1.7B)\n\n{tab}\n\nMean NF4/Q4_0 MSE ratio: block32 {np.mean(tot['a']):.2f}, block64 {np.mean(tot['b']):.2f}. Kurtosis: normal = 3.\n")
