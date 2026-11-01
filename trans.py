import os
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModelForCausalLM
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
