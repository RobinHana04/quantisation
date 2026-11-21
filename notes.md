# Notes

## Step 0: environment
- Hardware: MacBook Pro M3 Pro, 18 GB unified memory, Metal backend.
- Model: Qwen3-4B (BF16 ~7.5 GB) as main model; Qwen3-8B in BF16 (~16 GB) does not fit in the GPU-wired memory (~12-13 GB). Qwen3-1.7B for quick experiments.
- llama.cpp built from source (commit 690b87d) with -DGGML_METAL=ON in ./llama.cpp/build/bin
- Python 3.13 venv in ./.venv (torch 2.15, transformers, numpy, gguf, datasets, matplotlib, sentencepiece)
- Data: data/wiki.test.raw (perplexity), data/wiki.train.raw (imatrix calibration, step 4). The dataset id `wikitext` no longer works; use `Salesforce/wikitext`.

## Step 1: absmax basics
- One scale per group: `s = max|x|/(2^(b-1)-1)`. A single outlier determines the resolution of the whole group.
- int4 MSE on gate_proj (6144x2048): tensor 7.4e-4, channel 3.9e-5 (19x better), block 32 1.4e-5 (52x better than tensor).
- Biggest jump: tensor -> channel (rows have very different scales). After that the gain shrinks with each halving of the block.
- bpw rises with smaller blocks: `(b*B+16)/B`. Block 32 int4 = 4.5 bpw, block 16 = 5.0.
- Outlier block: max|w|=0.375 at median 0.0066 -> 29/32 weights become 0 in int4.

## Step 2: Q8_0 and Q4_0
- Q8_0: 34 bytes/block (fp16 d + 32 int8) = 8.5 bpw. Q4_0: 18 bytes (fp16 d + 16 bytes) = 4.5 bpw.
- Q4_0: d = max/-8 (max WITH sign), q = clamp(floor(x/d + 8.5), 0..15), x̂ = (q-8)*d. Uses all 16 levels, step max/8 instead of max/7: MSE 1.28x lower than symmetric int4 (1.12e-5 vs 1.42e-5).
- Nibble packing: byte j = elem j (low nibble) + elem j+16 (high nibble), so not neighbouring elements.
- Own implementation is byte-for-byte identical to gguf.quants (100%) for both formats.
- (s·q)·x = s·(q·x): integer sum per block, one float multiplication at the end.

## Step 3: GGUF and llama.cpp (Qwen3-4B)
- Conversion requires `sentencepiece` in the venv (the converter tries that tokenizer first).
- Q4_K_M = 4.95 bpw, not 4.5: half of attn_v and ffn_down (layers 0-3, every 3rd, last 6) and token_embd (tied with output) are Q6_K; norms stay F32.
- Quality: Q8_0/Q6_K/Q5_K_M are practically equal to F16 (ΔPPL <= 0.3%, KLD <= 0.017); Q4_K_M +1.9% PPL, KLD 0.062, 90% same top-1; Q3_K_M +22%; Q2_K +178% (unusable). The knee is between Q4 and Q3.
- Speed: tg ~ 1/size down to Q4_K_M (effective bandwidth ~120 GB/s = tg x size), pp stays flat (~480-560 t/s, compute-bound). Q3_K_M/Q2_K reach only ~78 GB/s: dequantization becomes the bottleneck.
- Pitfall: PPL(base) in KL mode (13.97) is not equal to the real F16 PPL (14.14). Use a separate F16 run as baseline.

## Step 4: imatrix
- imatrix = mean x[j]^2 per input column over calibration text (magnitude AND frequency of activations). quantize minimizes a weighted error instead of plain MSE.
- Built on 200 chunks of WikiText-2 train; evaluated on test (100 chunks). KLD reduction with imatrix: Q8_0 0%, Q6_K 19%, Q4_K_M 32%, Q3_K_M 35%, Q2_K 60% (PPL 39.2 -> 16.3). IQ2_XS (2.69 bpw, requires imatrix): PPL 19.98, KLD 0.76.
- The gain grows as bits shrink: fewer levels -> the choice of which weights get precision matters more.
- Note: Q4_K_M and Q6_K went up slightly in PPL (14.40 -> 14.69; 14.18 -> 14.20) while KLD dropped. PPL only looks at the correct token; KLD is more reliable here. Cause not investigated.

## Step 5A: NF4 vs uniform
- NF4 = 16 levels at quantiles of N(0,1), block first normalized by absmax. Only MSE on weights measured (no PPL; NF4 is not in llama.cpp).
- Block 32: NF4 ≈ Q4_0 (MSE ratio 0.99). Block 64: NF4 11% lower MSE (0.89), up to 17% for down_proj. Why it shrinks at block 32 is a hypothesis (smaller blocks are flatter distributed, Q4_0 uses the -8 level trick), not tested.
- Weights are not exactly normal: kurtosis 3.4-4.6 (normal = 3).
- Double quantization (scales to 8 bit): identical MSE, bpw 4.25 -> 4.13.
- NF4 block 64 + DQ (4.13 bpw) has ~10% higher MSE than Q4_0 block 32 (4.50 bpw) at 8% fewer bits.

## Step 5B: GPTQ-light
- GPTQ quantizes column by column and pushes the rounding error through H^-1 (H = X·Xt from calibration activations) onto the not-yet-rounded columns. The goal is minimal OUTPUT error ‖(W-Ŵ)X‖², not minimal weight error.
- On L10 gate_proj (int4, block 32): weight MSE 1.43x HIGHER than RTN (1.4e-5 -> 2.0e-5), but output error (test) 0.45x, so 2.2x lower. The RTN weight MSE matches step 1 exactly (1.424e-5), a sanity check.
- Overfit signal: GPTQ's relative output error is 1.35e-3 on train activations and 1.77e-3 on test (RTN: 3.87e-3 vs 3.92e-3, no gap). Same risk as imatrix: gained on calibration data, slightly less on new data.
- Limitations: one layer, per-layer error, no end-to-end PPL; activations come from the F32 model (no accumulated quantization error from earlier layers).
