#!/bin/bash
# Step 3: PPL + KL (against the F16 logits) and llama-bench per variant. Sequential, so the measurements do not disturb each other.
cd "$(dirname "$0")/.."
B=llama.cpp/build/bin
for t in Q8_0 Q6_K Q5_K_M Q4_K_M Q3_K_M Q2_K; do
  $B/llama-perplexity -m gguf/qwen3-4b-$t.gguf -f data/wiki.test.raw -c 512 --chunks 100 \
    --kl-divergence-base gguf/logits-f16.kld --kl-divergence > logs/kl-$t.log 2>&1
done
for t in f16 Q8_0 Q6_K Q5_K_M Q4_K_M Q3_K_M Q2_K; do
  $B/llama-bench -m gguf/qwen3-4b-$t.gguf -p 512 -n 128 -r 3 -o csv > logs/bench-$t.csv 2> logs/bench-$t.err
done
echo DONE > logs/step3.done
