from zipfile import ZipFile
import gdown
import requests
import os

U2NET_MODEL_URL = "https://drive.google.com/uc?id=1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ"
U2NETP_MODEL_URL = "https://drive.google.com/uc?id=1rbSTGKAE-MTxBYHd-51l2hMOQPT_7EPy"

U2NET_MODEL_PATH = "u2net.pth"
U2NETP_MODEL_PATH = "u2netp.pth"

if not os.path.exists(os.path.join(os.getcwd(), U2NET_MODEL_PATH)):
    _ = gdown.download(U2NET_MODEL_URL, U2NET_MODEL_PATH)

if not os.path.exists(os.path.join(os.getcwd(), U2NETP_MODEL_PATH)):
    _ = gdown.download(U2NETP_MODEL_URL, U2NETP_MODEL_PATH)
