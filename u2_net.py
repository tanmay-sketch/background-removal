import os
import requests
from PIL import Image
import numpy as np
from torchinfo import summary
import torchvision.transforms as T
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
