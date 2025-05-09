from transformers import LlamaForCausalLM

model_small = "meta-llama/Llama-2-7b-hf"
model_medium = "meta-llama/Llama-2-13b-hf"
model_large = "meta-llama/Llama-2-70b-hf"


model = LlamaForCausalLM.from_pretrained(
    model_medium,
    torch_dtype="auto",
    cache_dir="/scratch/p490-24-t/all_llamas",
    token="<INSERT_TOKEN_HERE>",
)
