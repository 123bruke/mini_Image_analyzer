import json, ast
from pathlib import Path

path = Path("/content/Qwen3_0_6B_CPU_LoRA_FineTuning.ipynb")
nb = json.loads(path.read_text(encoding="utf-8"))

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        src = "".join(cell["source"])
        if "from torch.nn.utils.rnn import pad_sequence" in src:
            src = src.replace("from torch.nn.utils.rnn import pad_sequence\n", "")
            src = src.replace("    TrainingArguments,\n    Trainer,\n", "")
            cell["source"] = src.splitlines(keepends=True)

        if src.startswith("def collate_examples(features):"):
            corrected = '''def collate_examples(features):
    # Build padding masks from each original sequence length.
    # Do not infer padding from token IDs because Qwen may use EOS as PAD.
    max_len = max(item["input_ids"].numel() for item in features)
    batch_size = len(features)

    input_ids = torch.full(
        (batch_size, max_len),
        tokenizer.pad_token_id,
        dtype=torch.long
    )
    labels = torch.full(
        (batch_size, max_len),
        -100,
        dtype=torch.long
    )
    attention_mask = torch.zeros(
        (batch_size, max_len),
        dtype=torch.long
    )

    for i, item in enumerate(features):
        length = item["input_ids"].numel()
        input_ids[i, :length] = item["input_ids"]
        labels[i, :length] = item["labels"]
        attention_mask[i, :length] = 1

    return {
        "input_ids": input_ids.to(DEVICE),
        "attention_mask": attention_mask.to(DEVICE),
        "labels": labels.to(DEVICE)
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
'''
            cell["source"] = corrected.splitlines(keepends=True)

# Syntax-check every Python cell after removing Colab magics.
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        src = "".join(cell["source"])
        lines = [line for line in src.splitlines() if not line.lstrip().startswith("%")]
        try:
            ast.parse("\n".join(lines))
        except SyntaxError as exc:
            raise RuntimeError(f"Syntax error in notebook code cell {i}: {exc}") from exc

path.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")
print("Notebook corrected and all Python code cells passed syntax checks.")
