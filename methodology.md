# Research methodology

## Research question
How much quality do you lose when you shrink the weights of Qwen3-4B, and what does that buy in memory and speed? In other words: the trade-off between **size, quality and speed**, per quantization type and per method.

## Setup
- **Model:** Qwen3-4B (main model), Qwen3-1.7B (fast experiments in steps 1 and 2)
- **Baseline:** F16 GGUF of Qwen3-4B. Everything is compared against it.
- **Hardware:** MacBook Pro M3 Pro, 18 GB unified memory, Metal backend
- **Evaluation data:** WikiText-2 test split
- **Calibration data (imatrix):** WikiText-2 train split. Never the test split, so the imatrix cannot "peek".

## Metrics

| Metric | What it tells you | Tool |
|---|---|---|
| MSE / max error | Distance between reconstructed weights x̂ and original x (one matrix, no model) | own NumPy code (step 1) |
| bpw (bits per weight) | Actual storage cost including scales: file size × 8 / number of weights | calculation |
| File size | Memory cost | `ls` |
| Perplexity (PPL) | exp(mean negative log-likelihood) on WikiText-2 test. Lower is better. | `llama-perplexity` |
| KL divergence | Deviation of the full next-token distribution from F16, averaged over all positions. More sensitive than PPL. Also "same top token %". | `llama-perplexity --kl-divergence-base` / `--kl-divergence` |
| pp (tokens/s) | Prompt processing, compute-bound | `llama-bench` |
| tg (tokens/s) | Token generation, memory-bandwidth-bound | `llama-bench` |

**Why PPL and KL?** PPL only looks at the probability of the correct token, and errors can cancel out. KL directly measures how different the behaviour is from the original.

## Variables
1. Block size and scale granularity: per tensor, channel, block 256/128/64/32/16 (step 1)
2. Quantization type: Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_M, Q2_K (step 3)
3. imatrix on or off, and calibration data (step 4)
4. Other methods: NF4, GPTQ (step 5, optional, reduced scope; see `summary.md`)

## Hypotheses (stated in advance, to be tested)
- Q8_0 is practically lossless; from about Q4 the degradation becomes noticeable; Q2_K is bad.
- Smaller quants are faster in tg, roughly proportional to size, but not in pp.
- imatrix helps mainly at low bits (Q2_K, IQ2_XS), hardly at Q8_0 and Q6_K.
- Smaller blocks give lower MSE, but cost more bpw because of scale overhead.

## Workflow and reproducibility
- One script per experiment in `experiments/`
- Results as tables in `results.md`
- Key lesson per step in `notes.md`
- Final summary in `summary.md`
- Everything in the project folder: venv in `.venv`, llama.cpp in `llama.cpp/` (commit `690b87d`); these and the model/data folders are gitignored
- For every experiment: first a prediction, then the measurement, then the comparison

## Limitations
- One model and one dataset: no general laws.
- PPL on WikiText says little about tasks such as reasoning or code.
- Speed only holds for this machine and fluctuates with temperature and background processes. Repeat measurements and report the mean.
- Qwen3-8B does not fit as an F16 baseline, so no KL divergence for that model.
