import torch
print(torch.cuda.is_available())        # True nếu có GPU
print(torch.cuda.device_count())        # số GPU
print(torch.cuda.current_device())      # chỉ số device
print(torch.cuda.get_device_name(0))    # tên GPU