from PIL import Image
import numpy as np
import torch
from torchinfo import summary
import torchvision.transforms as T
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from model import U2NET, U2NETP

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

u2net = U2NET(in_ch=3,out_ch=1)
u2netp = U2NETP(in_ch=3,out_ch=1)

print(f"{'U2NET Model Summary':=^90}")
print(summary(u2net, input_size=(1,3,320,320), device="cpu", depth=1, row_settings=["var_names"]))


