import torch

try:
    import torch_xla
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()

    print("TPU detected!")
    print("TPU device:", device)

except Exception as e:
    print("TPU unavailable:", e)
