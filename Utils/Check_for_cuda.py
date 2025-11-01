import torch
print(torch.__version__)        # PyTorch版本
print(torch.version.cuda)       # PyTorch编译时使用的CUDA版本
print(torch.cuda.is_available()) # 是否检测到GPU
print(torch.cuda.get_device_name(0))
