from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, get_scheduler
from torch.utils.data import DataLoader
from datasets import load_dataset, load_from_disk
from torch.cuda.amp import autocast, GradScaler
import torch

# Load dataset
data_dir = "../bookcorpus"
dataset = load_from_disk(data_dir)

print("Dataset loaded.")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def tokenize_data(example):
    return tokenizer(example['text'], padding='max_length', truncation=True)

tokenized_datasets = dataset.map(tokenize_data, batched=True, remove_columns=["text"])

# Load GPT model
config = GPT2Config(
    vocab_size=50257,       # Default vocabulary size for GPT-2
    n_embd=768,             # Hidden size (embedding size)
    n_layer=12,             # Number of layers
    n_head=12,              # Number of attention heads
    n_positions=1024        # Maximum sequence length
)

model = GPT2LMHeadModel(config).from_pretrained("gpt2").to("cuda")

def collate_fn(batch):
    return {
        "input_ids": torch.tensor([item["input_ids"] for item in batch]),
        "attention_mask": torch.tensor([item["attention_mask"] for item in batch]),
    }

# Training loop
train_dataloader = DataLoader(tokenized_datasets, batch_size=8, shuffle=True, collate_fn=collate_fn)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

num_training_steps = len(train_dataloader) * 3  # Assuming 3 epochs
lr_scheduler = get_scheduler(
    "linear", optimizer=optimizer, num_warmup_steps=500, num_training_steps=num_training_steps
)

print("Training started.")

scaler = GradScaler()
for epoch in range(3):  # 3 epochs
    model.train()
    for batch in train_dataloader:
        inputs = batch['input_ids'].to("cuda")
        attention_mask = batch["attention_mask"].to("cuda")

        optimizer.zero_grad()

        with autocast():
            outputs = model(inputs, attention_mask=attention_mask, labels=inputs)
            loss = outputs.loss

        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        lr_scheduler.step()

        print(f"Epoch {epoch}, Loss: {loss.item()}")

print('Finished Training')

model.save_pretrained("./gpt-model")
tokenizer.save_pretrained("./gpt-tokenizer")