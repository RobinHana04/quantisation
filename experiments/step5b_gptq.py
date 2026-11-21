"""Step 5B: GPTQ-light vs RTN on gate_proj of layer 10 (Qwen3-1.7B), int4 symmetric, block 32 (as in step 1)."""
import torch, numpy as np, time
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.manual_seed(0)
LAYER, GS, QMAX = 10, 32, 7
tok = AutoTokenizer.from_pretrained("models/Qwen3-1.7B")
model = AutoModelForCausalLM.from_pretrained("models/Qwen3-1.7B", dtype=torch.float32).to("mps").eval()
model.model.layers = model.model.layers[:LAYER + 1]            # layers 11+ are not needed
lin = model.model.layers[LAYER].mlp.gate_proj
W = lin.weight.detach().float().cpu().clone()                  # [6144, 2048]

def activations(path, n_chunks, skip=0):
    ids = tok(open(path).read()[:3_000_000], return_tensors="pt").input_ids[0]
    ids = ids[skip:skip + n_chunks * 512].view(n_chunks, 512)
    out = []
    h = lin.register_forward_hook(lambda m, i, o: out.append(i[0].detach().reshape(-1, i[0].shape[-1]).cpu()))
    with torch.no_grad():
        for b in range(0, n_chunks, 8): model(ids[b:b + 8].to("mps"))
    h.remove(); return torch.cat(out)
Xtr = activations("data/wiki.train.raw", 64); Xte = activations("data/wiki.test.raw", 32)
print("X train", tuple(Xtr.shape), "X test", tuple(Xte.shape))

def scale(g): return (g.abs().amax(1, keepdim=True) / QMAX).half().float().clamp_min(1e-10)
def rtn(W):
    Wh = torch.empty_like(W)
    for i in range(0, W.shape[1], GS):
        g = W[:, i:i + GS]; s = scale(g); Wh[:, i:i + GS] = s * torch.clamp(torch.round(g / s), -QMAX, QMAX)
    return Wh

def gptq(W, X, damp=0.01):
    W = W.clone(); n = W.shape[1]
    H = 2 * (X.T @ X) / X.shape[0]
    H += damp * H.diag().mean() * torch.eye(n)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    Q = torch.zeros_like(W)
    for j in range(n):
        if j % GS == 0: s = scale(W[:, j:j + GS])               # scale from the ALREADY updated weights
        w = W[:, j:j + 1]
        q = s * torch.clamp(torch.round(w / s), -QMAX, QMAX)
        Q[:, j:j + 1] = q
        err = (w - q) / Hinv[j, j]
        W[:, j + 1:] -= err @ Hinv[j:j + 1, j + 1:]               # push the error onto later columns
    return Q

def metrics(Wh, X):
    E = (W - Wh) @ X.T
    return float(((W - Wh) ** 2).mean()), float((E ** 2).mean()), float((E ** 2).sum() / ((W @ X.T) ** 2).sum())
t = time.time(); Wr = rtn(W); Wg = gptq(W, Xtr); print(f"gptq {time.time()-t:.0f}s")
rows = []
for name, Wh in (("RTN", Wr), ("GPTQ", Wg)):
    rows.append((name, metrics(Wh, Xte), metrics(Wh, Xtr)))
tab = ["| method | weight MSE | output error (test) | relative output error (test) | relative output error (train) |", "|---|---|---|---|---|"]
for n, te, tr in rows: tab.append(f"| {n} | {te[0]:.3e} | {te[1]:.3e} | {te[2]:.3e} | {tr[2]:.3e} |")
r, g = rows[0][1], rows[1][1]
tab = "\n".join(tab); print(tab)
print(f"GPTQ/RTN: weight-MSE {g[0]/r[0]:.2f}x, output error {g[1]/r[1]:.2f}x")
open("results.md", "a").write(f"\n## Step 5B: GPTQ-light vs RTN (L{LAYER} gate_proj, int4, block {GS}, calibration 32768 train tokens, damp 0.01)\n\n{tab}\n\nGPTQ/RTN: weight-MSE {g[0]/r[0]:.2f}x, output error (test) {g[1]/r[1]:.2f}x\n")
