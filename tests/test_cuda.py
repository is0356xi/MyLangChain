import torch

# CUDAが利用可能かどうかを確認
if torch.cuda.is_available():
    print("\n!!! CUDA is available !!!\n")
else:
    print("\n!!! CUDA is not available !!!\n")