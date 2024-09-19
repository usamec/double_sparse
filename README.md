## Dependencies

* `torch`: tested on v2.2.1
* `transformers`: tested on v4.35.2
* `datasets`: tested on v2.16.1

## Usage

We also provide LLaMA pruning script with the very same interface:

```
# Sparsify LLaMa with SparseGPT
python llama.py meta-llama/Llama-2-7b-hf c4 --sparsity 0.5
```
