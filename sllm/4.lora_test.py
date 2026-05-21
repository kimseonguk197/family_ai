import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
ADAPTER_PATH = "./llama32-3b-style-lora"

QUESTION = "Explain what Docker is."

def generate(model, tokenizer, question, max_new_tokens=180):
    messages = [
        {"role": "user", "content": question}
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            eos_token_id=tokenizer.eos_token_id
        )

    result = tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True)
    return result.strip()

def load_base():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    model.eval()
    return model, tokenizer

def load_lora():
    model, tokenizer = load_base()
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()
    return model, tokenizer

if __name__ == "__main__":
    print("=== BEFORE FT ===")
    base_model, tokenizer = load_base()
    print(generate(base_model, tokenizer, QUESTION))
    print()

    print("=== AFTER FT ===")
    lora_model, tokenizer = load_lora()
    print(generate(lora_model, tokenizer, QUESTION))