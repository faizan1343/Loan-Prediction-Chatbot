import os
import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from transformers import TrainingArguments, Trainer

# -----------------------
# Setup Logging
# -----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(r"E:\CSI_CB\logs\training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# -----------------------
# Config
# -----------------------
model_name = "meta-llama/Llama-3.2-1B-Instruct"
data_path = r"E:\CSI_CB\loan_qa_dataset_llama.csv"
output_dir = r"E:\CSI_CB\fine_tuned_model"
merged_model_dir = r"E:\CSI_CB\final_merged_model"
log_dir = r"E:\CSI_CB\logs"

# Create directories
os.makedirs(output_dir, exist_ok=True)
os.makedirs(merged_model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# -----------------------
# Tokenizer
# -----------------------
logger.info("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# -----------------------
# 4-bit Quantization Config
# -----------------------
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

# -----------------------
# Load Base Model (4-bit)
# -----------------------
logger.info("Loading base model in 4-bit...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto"
)

# Prepare for training
logger.info("Preparing model for training...")
model = prepare_model_for_kbit_training(model)
model.gradient_checkpointing_enable()
model.config.use_cache = False

# -----------------------
# Load Dataset & Tokenize
# -----------------------
logger.info("Loading and tokenizing dataset...")
dataset = load_dataset("csv", data_files=data_path)
dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)

def tokenize_function(examples):
    texts = [f"Question: {q.strip()}\nAnswer: {a.strip()}" for q, a in zip(examples["question"], examples["answer"])]
    model_inputs = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=512
    )
    # Mask padding tokens in labels to avoid unnecessary loss computation and reduce memory
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs.get("attention_mask")
    if attention_mask is not None:
        labels = []
        for ids_row, mask_row in zip(input_ids, attention_mask):
            labels.append([token_id if mask_val == 1 else -100 for token_id, mask_val in zip(ids_row, mask_row)])
        model_inputs["labels"] = labels
    else:
        model_inputs["labels"] = input_ids.copy()
    return model_inputs

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["question", "answer"]
)

# -----------------------
# LoRA Config
# -----------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

logger.info("Applying LoRA configuration...")
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# -----------------------
# Training Arguments
# -----------------------
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_dir=log_dir,
    logging_steps=10,
    save_steps=50,
    eval_strategy="steps",
    eval_steps=50,
    eval_accumulation_steps=1,
    group_by_length=True,
    fp16=True,
    save_total_limit=2,
    report_to="none"
)

# -----------------------
# Trainer
# -----------------------
logger.info("Initializing trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"]
)

# -----------------------
# Train
# -----------------------
logger.info("Starting training...")
trainer.train()

# -----------------------
# Save LoRA Adapter
# -----------------------
logger.info("Saving LoRA adapter and tokenizer...")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
logger.info(f"LoRA adapter and tokenizer saved at: {output_dir}")

# -----------------------
# Merge LoRA into Base Model
# -----------------------
logger.info("Merging LoRA weights into base model...")
try:
    # Reload base model on CPU for merge
    logger.info("Loading base model on CPU...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cpu"
    )

    # Load trained adapter
    logger.info("Loading LoRA adapters...")
    model_with_lora = PeftModel.from_pretrained(base_model, output_dir)

    # Merge weights
    logger.info("Merging weights...")
    merged_model = model_with_lora.merge_and_unload()

    # Save merged model
    logger.info("Saving merged model...")
    merged_model.save_pretrained(merged_model_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_model_dir)
    logger.info(f"Final merged model saved at: {merged_model_dir}")
except Exception as e:
    logger.error(f"Error during merging: {e}")
    raise

