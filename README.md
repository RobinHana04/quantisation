# LLM quantization study (Qwen3-4B, llama.cpp, Apple Silicon)

Hands-on study of LLM weight quantization: absmax basics, Q8_0/Q4_0 from scratch, GGUF K-quants, imatrix, NF4 and GPTQ.

- `summary.md`: results, chart, when to pick which type, key insights
- `methodology.md`: research question, metrics, hypotheses, limitations
- `results.md`: raw result tables; `notes.md`: lessons per step
- `experiments/`: one script per experiment; `logs/`: raw llama.cpp output

## Rebuilding the environment
The large folders are not in git (`.venv/`, `llama.cpp/`, `models/`, `gguf/`, `data/`). To rerun:

```bash
brew install cmake
git clone https://github.com/ggml-org/llama.cpp && git -C llama.cpp checkout 690b87d   # needs full history, not --depth 1
cmake -S llama.cpp -B llama.cpp/build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build llama.cpp/build -j --target llama-quantize llama-perplexity llama-bench llama-imatrix llama-cli
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python torch transformers numpy gguf datasets matplotlib huggingface_hub sentencepiece protobuf
.venv/bin/hf download Qwen/Qwen3-1.7B --local-dir models/Qwen3-1.7B     # ~3.8 GB
.venv/bin/hf download Qwen/Qwen3-4B   --local-dir models/Qwen3-4B       # ~7.5 GB
.venv/bin/python llama.cpp/convert_hf_to_gguf.py models/Qwen3-4B --outtype f16 --outfile gguf/qwen3-4b-f16.gguf
```

WikiText-2 (`Salesforce/wikitext`, config `wikitext-2-raw-v1`) is written to `data/wiki.{test,train}.raw` as one raw text file per split. Then run `experiments/step*.py|sh` in order (about 45 GB of disk for all GGUF variants and logits).
