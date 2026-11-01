from google.colab import drive
drive.mount("/content/drive")

WORK_DIR = Path("/content/qwen3_cpu_finetune")
OUTPUT_DIR = Path("/content/drive/MyDrive/qwen3-0.6b-cpu-lora")
WORK_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Working directory:", WORK_DIR)
print("Adapter output directory:", OUTPUT_DIR)
