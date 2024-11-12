import torch
print('CUDA版本:', torch.version.cuda)
print('Pytorch版本:', torch.__version__)
print('显卡是否可用:', '可用' if torch.cuda.is_available() else '不可用')
print('显卡数量:', torch.cuda.device_count())