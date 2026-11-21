"""Step 2: Q8_0 and Q4_0 from scratch, validated against the gguf package."""
import numpy as np, glob
from safetensors import safe_open
from gguf import quants, GGMLQuantizationType as T

NAME = "model.layers.10.mlp.gate_proj.weight"
for f in glob.glob("models/Qwen3-1.7B/*.safetensors"):
    with safe_open(f, "pt") as g:
        if NAME in g.keys():
            W = g.get_tensor(NAME).float().numpy()
B = W.reshape(-1, 32)                       # all blocks of 32

def rnd(v):                                  # C roundf: half away from zero
    return np.sign(v) * np.floor(np.abs(v) + 0.5)

# ---------- Q8_0: 2 bytes fp16 d + 32 int8 = 34 bytes ----------
def q8_0_quant(B):
    d = np.abs(B).max(1, keepdims=True) / 127
    id_ = np.where(d != 0, 1 / np.where(d == 0, 1, d), 0)
    q = rnd(B * id_).astype(np.int8)
    out = np.empty((len(B), 34), np.uint8)
    out[:, :2] = d.astype(np.float16).view(np.uint8)
    out[:, 2:] = q.view(np.uint8)
    return out
def q8_0_dequant(blk):
    d = blk[:, :2].copy().view(np.float16).astype(np.float32)
    return d * blk[:, 2:].copy().view(np.int8).astype(np.float32)

# ---------- Q4_0: 2 bytes fp16 d + 16 bytes (32 nibbles) = 18 bytes ----------
def q4_0_quant(B):
    imax = np.abs(B).argmax(1)
    mx = B[np.arange(len(B)), imax][:, None]          # value WITH sign
    d = mx / -8
    id_ = np.where(d != 0, 1 / np.where(d == 0, 1, d), 0)
    q = np.minimum(15, np.floor(B * id_ + 8.5)).astype(np.uint8)   # 0..15, offset 8
    packed = q[:, :16] | (q[:, 16:] << 4)             # byte j = elem j (low) + elem j+16 (high)
    out = np.empty((len(B), 18), np.uint8)
    out[:, :2] = d.astype(np.float16).view(np.uint8)
    out[:, 2:] = packed
    return out
def q4_0_dequant(blk):
    d = blk[:, :2].copy().view(np.float16).astype(np.float32)
    lo = (blk[:, 2:] & 0x0F).astype(np.int8) - 8
    hi = (blk[:, 2:] >> 4).astype(np.int8) - 8
    return d * np.concatenate([lo, hi], 1).astype(np.float32)

mse = lambda a, b: float(((a - b) ** 2).mean())
res = {}
for name, qf, df, t in (("Q8_0", q8_0_quant, q8_0_dequant, T.Q8_0), ("Q4_0", q4_0_quant, q4_0_dequant, T.Q4_0)):
    mine = qf(B)
    ref = quants.quantize(B, t)                       # official gguf implementation
    same_bytes = (mine == ref).mean()
    deq_mine = df(mine)
    deq_ref = quants.dequantize(ref, t)
    res[name] = (mine.shape[1], same_bytes, np.abs(deq_mine - deq_ref).max(), mse(B, deq_mine))
    print(f"{name}: {mine.shape[1]} bytes/block, {mine.shape[1]*8/32} bpw | bytes equal to gguf: {same_bytes:.4%} "
          f"| max dequant difference vs gguf: {res[name][2]:.2e} | MSE {res[name][3]:.3e}")

# symmetric int4 block 32 from step 1, for comparison
s = np.abs(B).max(1, keepdims=True) / 7
s = s.astype(np.float16).astype(np.float32)
mse_sym = mse(B, s * np.clip(np.round(B / s), -7, 7))
print(f"step-1 symmetric int4 block 32: MSE {mse_sym:.3e}  -> Q4_0 is {mse_sym/res['Q4_0'][3]:.2f}x better")

# nibble packing for the example block
ex = np.zeros((1, 32), np.float32); ex[0, :4] = [0.10, -0.20, 0.05, 0.70]; ex[0, 16] = -0.01
blk = q4_0_quant(ex)
q = np.minimum(15, np.floor(ex * (1 / (0.70 / -8)) + 8.5)).astype(int)[0]
print("\nPacking example: elem0 q=%d (%s), elem16 q=%d (%s) -> byte 0 = %s = 0x%02X" % (
    q[0], format(q[0], '04b'), q[16], format(q[16], '04b'), format(blk[0, 2], '08b'), blk[0, 2]))
print("  low nibble = elem0, high nibble = elem16 (not neighbours!)")

# (s*q).x = s*(q.x): integer dot product, scale applied only at the end
rng = np.random.default_rng(0)
x = rng.standard_normal(32).astype(np.float32)
w = B[1234]
bw = q8_0_quant(w[None]); bx = q8_0_quant(x[None])
sw = bw[0, :2].view(np.float16).astype(np.float32)[0]; sx = bx[0, :2].view(np.float16).astype(np.float32)[0]
qw = bw[0, 2:].view(np.int8).astype(np.int32); qx = bx[0, 2:].view(np.int8).astype(np.int32)
a = float(np.sum((sw * qw) * (sx * qx)))              # dequantize first, then float sum
b = float(sw * sx * int(np.dot(qw, qx)))              # integer sum, 1 float multiplication
print(f"\nDot product: float {a:.8f} vs integer-first {b:.8f} (unquantized): {w@x:.8f}")

open("results.md", "a").write(f"""
## Step 2: Q8_0 and Q4_0 from scratch ({NAME}, Qwen3-1.7B)

| format | bytes/block | bpw | bytes equal to gguf | max dequant difference | MSE |
|---|---|---|---|---|---|
| Q8_0 | {res['Q8_0'][0]} | 8.5 | {res['Q8_0'][1]:.4%} | {res['Q8_0'][2]:.1e} | {res['Q8_0'][3]:.3e} |
| Q4_0 | {res['Q4_0'][0]} | 4.5 | {res['Q4_0'][1]:.4%} | {res['Q4_0'][2]:.1e} | {res['Q4_0'][3]:.3e} |
| int4 symmetric (step 1, block 32) | - | 4.5 | - | - | {mse_sym:.3e} |
""")
