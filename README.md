# Mini Image Analyzer

A small Python repository for experimenting with machine-learning setup and model fine-tuning. Despite the repository name, the current scripts focus mostly on a CPU-based Qwen language-model training workflow—not on image analysis yet.

## What’s here

- `strter.py` — generates a Google Colab notebook for CPU-only Qwen3-0.6B LoRA fine-tuning.
- `eval.py` — checks and repairs notebook code cells before execution.
- `trans.py` — sets up model and training configuration.
- `model.py` — contains Google Drive and working-directory setup for Colab.
- `liser.py` — prints machine-learning package versions and checks CUDA availability.
- `cheak.py` — checks TPU availability.

## Quick start

The training workflow is designed for Google Colab. To generate the notebook:

```bash
python strter.py
```

The script writes `Qwen3_0_6B_CPU_LoRA_FineTuning.ipynb` in the Colab environment. Open the notebook, choose a CPU runtime, and run its cells in order.

To run the environment checks locally or in Colab:

```bash
python liser.py
python cheak.py
```

To validate notebook code cells:

```bash
python eval.py
```

## Requirements

Use Python 3.10 or newer. The notebook installs its training libraries when run:

```bash
pip install -U transformers peft accelerate datasets safetensors
```

PyTorch is also required. Google Colab is needed for the notebook’s Google Drive integration; the helper scripts may need changes to run outside Colab.

## Training notes

- The notebook uses the `Qwen/Qwen3-0.6B` base model and trains LoRA adapters in FP32 on CPU. This is LoRA fine-tuning, not QLoRA.
- CPU training can be very slow and may exceed a Colab session’s time or memory limits.
- The built-in examples are only a small pipeline test. Replace them with a larger, carefully checked dataset for meaningful results.
- The saved adapter is not a full copy of the base model.
- Fine-tuning does not guarantee accurate answers. Evaluate the model with separate test examples.

## Editing this README

You can update this file directly on GitHub: open `README.md`, select the pencil (Edit) button, make your changes, then select **Commit changes**. Alternatively, edit `README.md` in a text editor and commit the change with Git.

## License

No license file is currently included. Check with the repository owner before reusing this project.

## Disclaimer

This is an experimental and educational project. Its scripts may be incomplete and may require updates for your environment.
