import torch
print(torch.version.cuda)        # Versión de CUDA que PyTorch detecta
print(torch.cuda.is_available()) # True si detecta GPU
print(torch.cuda.get_device_name(0))  # Nombre de la GPU
