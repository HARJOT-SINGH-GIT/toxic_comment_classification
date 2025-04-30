# toxic_comment_classification.py

import os
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from transformers import DataCollatorWithPadding
from optimum.onnxruntime import ORTTrainer, ORTTrainingArguments
from huggingface_hub import login


hf_token = os.getenv("HF_TOKEN_KEY")
login(token = hf_token)
# Check GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load dataset 
dataset = load_dataset("sms_spam", split="train") 
# dataset = pd.read_csv(r"D:\project\NLP_assignment\archive\train.csv") 

# Preprocess the dataset
from transformers import DistilBertTokenizer, TFDistilBertModel
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def preprocess_function(examples):
    return tokenizer(examples['sms'], truncation=True)

encoded_dataset = dataset.map(preprocess_function)

# Binary labels (toxic=1, non-toxic=0) 
def label_encode(example):
    example['labels'] = 1 if example['label'] == 1 else 0
    return example

encoded_dataset = encoded_dataset.map(label_encode)

# Load base model
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", 
    num_labels=2
)

# Apply LoRA (Low Rank Adaptation)
# lora_config = LoraConfig(
#     task_type=TaskType.SEQ_CLS, 
#     r=8, 
#     lora_alpha=16, 
#     lora_dropout=0.1,
#     bias="none"
# )
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    target_modules=["q_lin", "v_lin"]  # Important for DistilBERT
)
model = get_peft_model(model, lora_config)
print("LoRA adapters added.")

# Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Training arguments
training_args = TrainingArguments(
    output_dir="./lora-toxic-classifier",
    eval_strategy="epoch",
    learning_rate=2e-4,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=1,
    load_best_model_at_end=True,
    save_strategy="epoch",
    logging_dir='./logs',
    logging_steps=10,
    push_to_hub=False,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset,
    eval_dataset=encoded_dataset,  
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Train
trainer.train()

# Save the LoRA fine-tuned model
model.save_pretrained("./lora-toxic-classifier")
tokenizer.save_pretrained("./lora-toxic-classifier")

print("Model fine-tuned and saved.")


from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel

# Load base model and tokenizer
base_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Load LoRA weights
model = PeftModel.from_pretrained(base_model, "lora-toxic-classifier")

# Merge LoRA into base model
model = model.merge_and_unload()

# Save merged model + tokenizer
save_path = "lora-toxic-classifier-merged"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)


# Convert to ONNX for faster inference 
from transformers.onnx import export
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers.onnx import OnnxConfig
from pathlib import Path
import torch

model_path = "lora-toxic-classifier-merged"

model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

class CustomOnnxConfig(OnnxConfig):
    @property
    def inputs(self):
        return {
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
        }

onnx_config = CustomOnnxConfig(model.config)

export(
    preprocessor=tokenizer,
    model=model,
    config=onnx_config,
    opset=17,
    output=Path("lora-toxic-classifier.onnx")
)
print("Model exported to ONNX.")



