import os
import gc
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, PeftModel
from trl import SFTTrainer, SFTConfig
from huggingface_hub import login

# =========================================================
# 0. 기본 설정
# =========================================================
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
DATA_PATH = "train_style.jsonl"

ADAPTER_DIR = "./llama32-3b-style-lora"
MERGED_DIR = "./llama32-3b-style-merged"

DEVICE = "cpu"
DTYPE = torch.float32  # CPU에서는 float32가 가장 안전

# CPU 스레드 수 제한
# 너무 높게 잡으면 오히려 느려질 수 있어 보통 4~8 정도가 무난
torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))


# =========================================================
# 1. HF 로그인
# =========================================================
def hf_login():
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
        print("[OK] Hugging Face login successful.")
    else:
        print("[WARN] HF_TOKEN environment variable not found. Private model 접근이 필요하면 설정하세요.")


# =========================================================
# 2. 데이터 전처리
# =========================================================
def build_text(example, tokenizer):
    """
    instruction / response 형식의 jsonl 데이터를
    Llama chat template 문자열로 변환
    """
    messages = [
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["response"]},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text": text}


def load_and_prepare_dataset(tokenizer):
    dataset = load_dataset("json", data_files=DATA_PATH)["train"]

    # 원본 컬럼 제거 후 text 컬럼만 유지
    dataset = dataset.map(
        lambda x: build_text(x, tokenizer),
        remove_columns=dataset.column_names
    )
    return dataset


# =========================================================
# 3. 모델 / 토크나이저 로드
# =========================================================
def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

    # pad_token 미설정 방지
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"
    return tokenizer


def load_base_model():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        low_cpu_mem_usage=True
    )
    model.to(DEVICE)

    # 학습 메모리 절약
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    return model


# =========================================================
# 4. LoRA 학습
# =========================================================
def train_lora():
    print("[INFO] Loading tokenizer...")
    tokenizer = load_tokenizer()

    print("[INFO] Loading base model...")
    model = load_base_model()

    print("[INFO] Loading dataset...")
    dataset = load_and_prepare_dataset(tokenizer)

    # CPU 환경에서는 attention 계열만 거는 편이 더 현실적
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    # 최신 TRL 방식: SFTConfig 사용
    sft_config = SFTConfig(
        output_dir=ADAPTER_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=1e-4,
        num_train_epochs=2,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="no",
        max_length=192,   # 3B + CPU 환경 고려
        fp16=False,
        bf16=False,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        report_to="none",
        dataloader_num_workers=0,   # Windows + CPU 안정성
        remove_unused_columns=False,
        packing=False,              # CPU에서는 단순하게 유지
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=sft_config,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    print("[INFO] Training started...")
    trainer.train()

    print("[INFO] Saving LoRA adapter...")
    trainer.model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)

    print(f"[OK] adapter saved to: {ADAPTER_DIR}")

    # 메모리 정리
    del trainer
    del model
    gc.collect()


# =========================================================
# 5. LoRA 병합
# =========================================================
def merge_lora_to_base():
    print("[INFO] Loading base model for merge...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        low_cpu_mem_usage=True
    )
    base_model.to(DEVICE)

    print("[INFO] Loading trained adapter...")
    peft_model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_DIR
    )

    print("[INFO] Merging adapter into base model...")
    merged_model = peft_model.merge_and_unload()

    os.makedirs(MERGED_DIR, exist_ok=True)

    print("[INFO] Saving merged model...")
    merged_model.save_pretrained(MERGED_DIR, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(MERGED_DIR)

    print(f"[OK] merged model saved to: {MERGED_DIR}")

    # 메모리 정리
    del base_model
    del peft_model
    del merged_model
    gc.collect()


# =========================================================
# 6. 간단 추론 테스트
# =========================================================
def test_merged_model(prompt: str):
    print("[INFO] Loading merged model for inference...")
    tokenizer = AutoTokenizer.from_pretrained(MERGED_DIR, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MERGED_DIR,
        torch_dtype=DTYPE,
        low_cpu_mem_usage=True
    )
    model.to(DEVICE)
    model.eval()

    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(input_text, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n[TEST OUTPUT]")
    print(result)


# =========================================================
# 7. 메인
# =========================================================
def main():
    hf_login()
    train_lora()
    merge_lora_to_base()

    # 필요하면 테스트
    # test_merged_model("안녕하세요. 자기소개를 해주세요.")


if __name__ == "__main__":
    main()