"""Step 1: symmetric absmax quantization, per tensor / channel / block."""
import numpy as np, torch
from safetensors import safe_open
import glob

NAME = "model.layers.10.mlp.gate_proj.weight"
for f in glob.glob("models/Qwen3-1.7B/*.safetensors"):
    with safe_open(f, "pt") as g:
        if NAME in g.keys():
            W = g.get_tensor(NAME).float().numpy()
print("shape", W.shape, "std", W.std(), "max|w|", np.abs(W).max())

def quant(x, bits, axis_groups):
    """x: array whose last axis is the group. Returns x̂."""
    qmax = 2 ** (bits - 1) - 1
    s = np.abs(x).max(axis=-1, keepdims=True) / qmax
    s = np.where(s == 0, 1, s)
    s = s.astype(np.float16).astype(np.float32)   # the scale is stored as fp16
    q = np.clip(np.round(x / s), -qmax, qmax)
    return s * q

def run(W, bits, gran):
    if gran == "tensor":   g = W.reshape(1, -1)
    elif gran == "channel": g = W
    else:                  g = W.reshape(-1, int(gran))
    Wh = quant(g, bits, None).reshape(W.shape)
    e = W - Wh
    return (e ** 2).mean(), np.abs(e).max()

def bpw(bits, gran, W):
    n = W.size
    groups = 1 if gran == "tensor" else (W.shape[0] if gran == "channel" else n // int(gran))
    return (bits * n + 16 * groups) / n

rows = []
for bits in (8, 4):
    for gran in ("tensor", "channel", 256, 128, 64, 32, 16):
        mse, mx = run(W, bits, gran)
        rows.append((bits, gran, mse, mx, bpw(bits, gran, W)))

out = ["| bits | granularity | MSE | max error | bpw |", "|---|---|---|---|---|"]
for b, g, m, mx, p in rows:
    out.append(f"| int{b} | {g} | {m:.3e} | {mx:.4f} | {p:.4f} |")
table = "\n".join(out)
print(table)

# outlier effect: block of 32 with the largest max/median ratio
B = W.reshape(-1, 32)
ratio = np.abs(B).max(1) / (np.median(np.abs(B), 1) + 1e-9)
i = int(np.argmax(ratio)); blk = B[i]
s = np.abs(blk).max() / 7
q = np.round(blk / s)
typ = B[len(B) // 2]; st = np.abs(typ).max() / 7; qt = np.round(typ / st)
msg = (f"\nOutlier block (index {i}): max|w|={np.abs(blk).max():.4f}, median|w|={np.median(np.abs(blk)):.4f}, "
       f"s={s:.4f}\n  {int((q==0).sum())}/32 weights become 0 (int4); values q: {q.astype(int).tolist()}"
       f"\n  Typical block: s={st:.4f}, {int((qt==0).sum())}/32 zeros, q range {qt.min():.0f}..{qt.max():.0f}"
       f"\n  Fraction of zeros over all blocks: {(np.round(B/(np.abs(B).max(1,keepdims=True)/7))==0).mean():.2%}")
print(msg)
open("results.md", "a").write(f"\n## Step 1: absmax on {NAME} (Qwen3-1.7B, shape {W.shape})\n\n{table}\n\n```{msg}\n```\n")
