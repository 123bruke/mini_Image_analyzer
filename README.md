# Mini Image Analyzer

A lightweight Python experiment repository for exploring small-scale AI/ML workflows, model fine-tuning, and local validation in a CPU-only environment.

This project currently contains a set of helper scripts and experimentation files centered around:

- CPU-only model setup and environment checks
- Qwen-style LLM loading and training preparation
- LoRA fine-tuning experiments
- Notebook generation and validation
- Dataset preparation for instruction-following tasks

Although the repository is named `mini_Image_analyzer`, the active code in this branch is primarily focused on LLM experimentation and fine-tuning rather than a production image-analysis application.

## Project purpose

The repository is intended as a compact research sandbox for:

- testing model environments before training
- preparing small instruction datasets
- validating notebook-based workflows
- checking CPU compatibility and dependencies
- experimenting with LoRA-based fine-tuning on small models

## Repository structure

- `strter.py` — builds a notebook-style training pipeline for CPU-only fine-tuning
- `trans.py` — initializes model/training configuration and environment settings
- `model.py` — sets up the working directory and Drive output folder for Colab
- `liser.py` — prints installed ML package versions and checks CUDA availability
- `cheak.py` — checks whether TPU support is available
- `eval.py` — validates and repairs notebook code cells before execution

## Main workflow

The project follows a typical small-model experimentation flow:

1. Install required ML libraries
2. Set up a CPU-only runtime for training
3. Prepare a small JSONL instruction dataset
4. Load a pretrained model such as `Qwen/Qwen3-0.6B`
5. Add LoRA adapters
6. Train for a short period on a limited dataset
7. Save the adapter and tokenizer
8. Evaluate and compare outputs

## Dependencies

This project relies on:

- Python 3.10+
- PyTorch
- Transformers
- PEFT
- Datasets
- Accelerate
- Google Colab utilities (for notebook-based execution)

Typical install command:

```bash
pip install -U transformers peft accelerate datasets safetensors
```

## Example usage

Run the scripts in a suitable environment such as Google Colab or a local workstation with Python installed.

### Check environment

```bash
python liser.py
```

### Check TPU availability

```bash
python cheak.py
```

### Create a notebook-based training workflow

```bash
python strter.py
```

### Validate notebook cells

```bash
python eval.py
```

## Notes

- The project is designed for experimentation and educational use.
- CPU training is significantly slower than GPU training.
- Small demo datasets are useful for smoke tests, but they do not produce a strong domain expert model.
- Fine-tuning with LoRA is a lightweight adaptation technique, but it does not replace careful data curation or evaluation.

## Recommended next steps

- Replace the starter dataset with a real task-specific dataset
- Use a stronger validation set for evaluating quality
- Consider RAG for knowledge-heavy tasks that require up-to-date external context
- Expand the project into a real image-analysis or inspection tool if that is the intended application scope

## License

This project does not currently include an explicit license file. Please check the repository settings or contact the repository owner before using it in a production or commercial context.

## Disclaimer

This repository is primarily a research and experimentation workspace. It may be incomplete, exploratory, or adapted for educational use.
