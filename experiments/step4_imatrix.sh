#!/bin/bash
# Step 4: imatrix (calibration = WikiText-2 TRAIN), quantize with imatrix, PPL+KL on TEST.
cd "$(dirname "$0")/.."
B=llama.cpp/build/bin
$B/llama-imatrix -m gguf/qwen3-4b-f16.gguf -f data/wiki.train.raw -c 512 --chunks 200 -o gguf/imatrix-train.gguf > logs/imatrix.log 2>&1
for t in Q8_0 Q6_K Q4_K_M Q3_K_M Q2_K IQ2_XS; do
  $B/llama-quantize --imatrix gguf/imatrix-train.gguf gguf/qwen3-4b-f16.gguf gguf/qwen3-4b-$t-im.gguf $t > logs/quantize-$t-im.log 2>&1
  $B/llama-perplexity -m gguf/qwen3-4b-$t-im.gguf -f data/wiki.test.raw -c 512 --chunks 100 \
    --kl-divergence-base gguf/logits-f16.kld --kl-divergence > logs/kl-$t-im.log 2>&1
done
echo DONE > logs/step4.done
