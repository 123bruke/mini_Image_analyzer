import json
from pathlib import Path

nb_cells = []

def md(text):
    nb_cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True)
    })

def code(text):
    nb_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True)
    })

md("""# CPU-only fine-tuning: Qwen3-0.6B with LoRA

This notebook fine-tunes **Qwen/Qwen3-0.6B** using a **CPU-only Google Colab runtime**.

- Method: supervised fine-tuning (SFT) with LoRA adapters
- Precision: FP32
- Device: CPU
- Libraries: PyTorch, Transformers, PEFT, Datasets
- Output: a small LoRA adapter, not a complete copy of the base model

**Important:** This is ordinary LoRA fine-tuning, not QLoRA. QLoRA requires a quantized base model; this notebook deliberately uses FP32 CPU training.

CPU training is slow. Start with the small demonstration dataset and one epoch. A tiny demo dataset proves that the pipeline runs; it does not produce a strong domain expert. Replace or expand it with many accurate, diverse examples for useful specialization.
""")

md("""## 1. Select a CPU runtime

In Colab, select **Runtime → Change runtime type → CPU**. Then run the cells in order.

The notebook explicitly sets `use_cpu=True` for training and loads the model on CPU. Colab's available RAM and session duration can vary, so successful execution is not guaranteed on every runtime allocation.
""")

code("""# Install the training libraries.
%pip -q install -U transformers peft accelerate datasets safetensors
""")

code("""import os
import json
import random
import math
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

# Force this experiment to use CPU.
DEVICE = torch.device("cpu")
torch.set_num_threads(max(1, min(4, os.cpu_count() or 2)))

MODEL_ID = "Qwen/Qwen3-0.6B"
MAX_LENGTH = 384
BATCH_SIZE = 1
GRAD_ACCUMULATION = 4
EPOCHS = 1
LEARNING_RATE = 1e-4

print("PyTorch:", torch.__version__)
print("Device:", DEVICE)
print("CPU threads:", torch.get_num_threads())
print("CUDA available (not used):", torch.cuda.is_available())
""")

md("""## 2. Keep the result in Google Drive

Mounting Drive lets the adapter survive after the Colab session ends. The base model is downloaded from Hugging Face into the Colab cache.
""")

code("""from google.colab import drive
drive.mount("/content/drive")

WORK_DIR = Path("/content/qwen3_cpu_finetune")
OUTPUT_DIR = Path("/content/drive/MyDrive/qwen3-0.6b-cpu-lora")
WORK_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Working directory:", WORK_DIR)
print("Adapter output directory:", OUTPUT_DIR)
""")

md("""## 3. Create a starter instruction dataset

The examples below specialize the assistant in AI/ML and knowledge-graph engineering concepts. They are only a small smoke-test dataset.

For meaningful fine-tuning, prepare at least hundreds of high-quality examples. Each record should contain a clear user question and a correct assistant answer. Avoid duplicate, contradictory, or unverified answers.
""")

code("""starter_examples = [
    ("What is machine learning?",
     "Machine learning is a field of artificial intelligence in which a model learns patterns from examples and uses those patterns to make predictions or decisions."),
    ("What is supervised learning?",
     "Supervised learning trains a model using examples that contain inputs and known target labels. The model learns a mapping from inputs to targets."),
    ("What is a classification problem?",
     "Classification predicts a category or class, such as spam versus not spam or one disease category versus another."),
    ("What is regression?",
     "Regression predicts a numeric value, such as a house price, temperature, or delivery time."),
    ("What is a neural network?",
     "A neural network is a machine-learning model made of connected layers of mathematical units. It learns by adjusting weights to reduce prediction error."),
    ("What is backpropagation?",
     "Backpropagation calculates gradients of the loss with respect to trainable parameters. An optimizer uses those gradients to update the parameters."),
    ("What is overfitting?",
     "Overfitting occurs when a model performs very well on its training examples but poorly on new examples. It often indicates that the model has learned details that do not generalize."),
    ("What is an LLM?",
     "A large language model is a neural network trained on large amounts of text to predict and generate token sequences. It can be adapted for tasks such as question answering, summarization, and coding."),
    ("What is a Transformer?",
     "A Transformer is a neural-network architecture that uses attention mechanisms to process relationships between tokens. It is widely used in modern language models."),
    ("What is an embedding?",
     "An embedding is a numeric vector representing an item such as a word, sentence, image, or document. Similar items can have vectors that are close in the embedding space."),
    ("What is retrieval-augmented generation?",
     "Retrieval-augmented generation, or RAG, retrieves relevant information from an external source and supplies it to a language model as context for generating an answer."),
    ("What does LlamaIndex do?",
     "LlamaIndex is a framework for connecting language models to external data. It supports document ingestion, indexing, retrieval, and query workflows."),
    ("What is a vector database?",
     "A vector database stores and searches vector embeddings. It can retrieve items whose vectors are similar to a query vector."),
    ("What is a knowledge graph?",
     "A knowledge graph represents information as entities and relationships. For example, a Gene entity can be connected to a Disease entity by a relationship."),
    ("What is Neo4j?",
     "Neo4j is a graph database. It stores data as nodes, relationships, and properties and supports graph queries using Cypher."),
    ("What is BioCypher?",
     "BioCypher is a framework for building biological knowledge graphs. It helps map biological source data to standardized entities and relationships."),
    ("What is a BioCypher adapter?",
     "A BioCypher adapter processes data from a particular source and maps it to standardized entities and relationships that can be used to construct a knowledge graph."),
    ("What is MeTTa?",
     "MeTTa is a symbolic and logical programming language used in the OpenCog Hyperon ecosystem. It can represent facts and rules and perform reasoning over them."),
    ("What is model evaluation?",
     "Model evaluation measures how well a trained model performs on data that was not used to fit its parameters. The metrics should match the task."),
    ("What is the F1 score?",
     "The F1 score is the harmonic mean of precision and recall. It is useful when both false positives and false negatives matter."),
    ("What is data leakage?",
     "Data leakage occurs when information unavailable at prediction time enters training or evaluation. It can make reported model performance misleadingly high."),
    ("What is LoRA fine-tuning?",
     "LoRA adds small trainable low-rank matrices to selected layers of a pretrained model. The original model weights are frozen while the adapter parameters are trained."),
    ("What is the difference between fine-tuning and RAG?",
     "Fine-tuning changes trainable model parameters using examples. RAG retrieves external information at answer time and supplies it as context without changing the model weights."),
    ("What is a train-validation split?",
     "A train-validation split divides examples into a training portion used to update the model and a validation portion used to monitor generalization during development.")
]

SYSTEM_PROMPT = (
    "You are a careful AI engineering tutor. "
    "Explain technical topics accurately in clear, simple language. "
    "Do not invent details when you are unsure."
)

records = [
    {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ]
    }
    for question, answer in starter_examples
]

DATA_FILE = WORK_DIR / "train.jsonl"
with DATA_FILE.open("w", encoding="utf-8") as f:
    for row in records:
        f.write(json.dumps(row, ensure_ascii=False) + "\\n")

print(f"Created {len(records)} starter examples at {DATA_FILE}")
""")

md("""### Optional: use your own JSONL dataset

To use your own data, upload a `.jsonl` file with one conversation per line. Each line must look like this:

```json
{"messages":[{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":"What is RAG?"},{"role":"assistant","content":"RAG retrieves relevant external information and provides it to an LLM as context."}]}
```

After uploading, set `DATA_FILE` to the uploaded file's path. Keep one user request followed by one assistant response per record for this notebook.
""")

code("""# Optional upload:
# from google.colab import files
# uploaded = files.upload()
# DATA_FILE = Path(next(iter(uploaded.keys())))
#
# Check your records:
with DATA_FILE.open(encoding="utf-8") as f:
    records = [json.loads(line) for line in f if line.strip()]

assert len(records) >= 5, "Please provide at least 5 examples."
for i, row in enumerate(records):
    assert "messages" in row and len(row["messages"]) >= 2, f"Invalid record {i}"
    assert row["messages"][-1]["role"] == "assistant", f"Last message must be assistant in record {i}"

print("Dataset records:", len(records))
print("First example:", records[0]["messages"])
""")

md("""## 4. Load the tokenizer and pretrained LLM on CPU

The base model remains in FP32. This avoids GPU-only 4-bit quantization steps and makes the training setup genuinely CPU-based.
""")

code("""tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True
)
model.to(DEVICE)
model.config.use_cache = False

print("Loaded:", MODEL_ID)
print("Parameter count:", f"{sum(p.numel() for p in model.parameters()):,}")
""")

md("""## 5. Check the original model before fine-tuning

This gives us a baseline response for comparison. The model is still the original pretrained model at this point.
""")

code("""def ask_model(model_to_use, question, max_new_tokens=100):
    model_to_use.eval()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]
    prompt_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        output_ids = model_to_use.generate(
            input_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    new_tokens = output_ids[0, prompt_ids.shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

baseline_question = "What is a BioCypher adapter?"
print("BASE MODEL RESPONSE:")
print(ask_model(model, baseline_question))
""")

md("""## 6. Split data and prepare assistant-only training labels

The notebook masks the system and user prompt tokens with `-100`. The loss is therefore calculated on the assistant answer tokens, rather than asking the model to learn to reproduce the user prompt.
""")

code("""random.shuffle(records)
split_at = max(1, int(len(records) * 0.9))
train_records = records[:split_at]
eval_records = records[split_at:]

if len(eval_records) == 0:
    eval_records = train_records[-1:]
    train_records = train_records[:-1]

def encode_record(record):
    messages = record["messages"]

    full_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=False
    )

    prompt_ids = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False
    )

    full_ids = full_ids[:MAX_LENGTH]
    prompt_length = min(len(prompt_ids), len(full_ids))

    # Only train on the assistant answer.
    labels = [-100] * prompt_length + full_ids[prompt_length:]

    if not any(label != -100 for label in labels):
        raise ValueError("An example has no assistant answer tokens after truncation. Increase MAX_LENGTH.")

    return {
        "input_ids": torch.tensor(full_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long)
    }

train_features = [encode_record(row) for row in train_records]
eval_features = [encode_record(row) for row in eval_records]

print("Train examples:", len(train_features))
print("Validation examples:", len(eval_features))
print("Maximum sequence length:", MAX_LENGTH)
""")

md("""## 7. Add LoRA adapters

Only the LoRA adapter parameters will be trainable. The pretrained base-model parameters remain frozen.
""")

code("""lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ]
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.3f}%)")
""")

md("""## 8. Create CPU data loaders and train

This uses a small batch size and gradient accumulation to keep the experiment modest. Expect CPU training to be slow, especially with longer sequences or larger datasets.
""")

code("""def collate_examples(features):
    input_ids = [item["input_ids"] for item in features]
    labels = [item["labels"] for item in features]

    padded_ids = pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=tokenizer.pad_token_id
    )
    padded_labels = pad_sequence(
        labels,
        batch_first=True,
        padding_value=-100
    )
    attention_mask = (padded_ids != tokenizer.pad_token_id).long()

    return {
        "input_ids": padded_ids.to(DEVICE),
        "attention_mask": attention_mask.to(DEVICE),
        "labels": padded_labels.to(DEVICE)
    }

train_loader = DataLoader(
    train_features,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_examples,
    num_workers=0
)

eval_loader = DataLoader(
    eval_features,
    batch_size=1,
    shuffle=False,
    collate_fn=collate_examples,
    num_workers=0
)

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=LEARNING_RATE
)

def evaluate_loss():
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in eval_loader:
            outputs = model(**batch)
            losses.append(float(outputs.loss.detach().cpu()))
    model.train()
    return sum(losses) / max(1, len(losses))

history = []
optimizer.zero_grad(set_to_none=True)

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(train_loader, start=1):
        outputs = model(**batch)
        raw_loss = outputs.loss
        (raw_loss / GRAD_ACCUMULATION).backward()
        running_loss += float(raw_loss.detach().cpu())

        should_update = (
            step % GRAD_ACCUMULATION == 0
            or step == len(train_loader)
        )

        if should_update:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=1.0
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if step % 5 == 0 or step == len(train_loader):
            print(
                f"Epoch {epoch + 1}/{EPOCHS} | "
                f"Step {step}/{len(train_loader)} | "
                f"Average training loss: {running_loss / step:.4f}"
            )

    train_loss = running_loss / max(1, len(train_loader))
    val_loss = evaluate_loss()

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "validation_loss": val_loss
    })

    print(
        f"Epoch {epoch + 1} complete | "
        f"train loss={train_loss:.4f} | "
        f"validation loss={val_loss:.4f}"
    )

print("Training loop finished.")
""")

md("""## 9. Save the fine-tuned adapter and tokenizer

The adapter folder does not contain the full base LLM. Keep the model ID `Qwen/Qwen3-0.6B` with the adapter so it can be loaded again later.
""")

code("""model.save_pretrained(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))

history_path = OUTPUT_DIR / "training_history.json"
with history_path.open("w", encoding="utf-8") as f:
    json.dump(history, f, indent=2)

print("Adapter saved to:", OUTPUT_DIR)
print("Files:", [p.name for p in OUTPUT_DIR.iterdir()])
""")

md("""## 10. Compare the fine-tuned model with the original model

Use the same questions for both. Improvement must be judged by correctness and usefulness, not by a lower training loss alone.
""")

code("""test_questions = [
    "What is a BioCypher adapter?",
    "Explain the difference between fine-tuning and RAG.",
    "What is backpropagation?"
]

for question in test_questions:
    print("\\n" + "=" * 80)
    print("QUESTION:", question)
    print("\\nFINE-TUNED RESPONSE:")
    print(ask_model(model, question, max_new_tokens=120))
""")

md("""## 11. Download a copy of the adapter

The adapter is already saved in Google Drive. This cell also creates a ZIP file for download.
""")

code("""import shutil
zip_path = shutil.make_archive(
    "/content/qwen3-0.6b-cpu-lora",
    "zip",
    root_dir=str(OUTPUT_DIR)
)

print("ZIP created:", zip_path)

from google.colab import files
files.download(zip_path)
""")

md("""## What this experiment produces

- A Qwen3-0.6B LoRA adapter trained on your JSONL examples
- Training and validation loss measurements
- Example outputs from the adapted model
- A ZIP archive containing the adapter and tokenizer files

## Important limitations

1. CPU training is much slower than GPU training. A long dataset may take a very long time or exceed the Colab session limit.
2. The built-in examples are a pipeline test, not enough to create a dependable expert assistant.
3. Validation loss on a tiny holdout set is noisy. Evaluate with a separate, carefully written test set as well.
4. Fine-tuning does not guarantee factual correctness or add a live document search capability. Use RAG for changing or large external knowledge sources.
5. This is LoRA FP32 training, not 4-bit QLoRA.
""")

notebook = {
    "cells": nb_cells,
    "metadata": {
        "colab": {
            "name": "Qwen3_0_6B_CPU_LoRA_FineTuning.ipynb",
            "provenance": []
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

out_path = Path("/content/Qwen3_0_6B_CPU_LoRA_FineTuning.ipynb")
out_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Created notebook: {out_path}")
