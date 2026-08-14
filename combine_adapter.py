from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base_model_id = "google/gemma-3-12b-it"
adapter_path = "dpo_model_gemma-3-12b-it"

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, adapter_path)
tokenizer = AutoTokenizer.from_pretrained(adapter_path)
merged_model = model.merge_and_unload()
merged_model.save_pretrained("results/dpo_model_gemma-3-12b-it-merged")
tokenizer.save_pretrained("results/dpo_model_gemma-3-12b-it-merged")
