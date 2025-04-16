import os 
from PIL import Image
import numpy as np
import torch
from torchinfo import summary
import torchvision.transforms as T
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from model import U2NET, U2NETP

U2NET_MODEL_PATH = './model/u2net.pth'
U2NETP_MODEL_PATH = './model/u2netp.pth'
TEST_IMAGE_DIR = '../subset/'
OUTPUT_PATH = '../subset_u2net/'

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

u2net = U2NET(in_ch=3,out_ch=1)
u2netp = U2NETP(in_ch=3,out_ch=1)

# print(f"{'U2NET Model Summary':=^90}")
# print(summary(u2net, input_size=(1,3,320,320), device="cpu", depth=1, row_settings=["var_names"]))

def load_model(model, model_path, device):
    model.load_state_dict(torch.load(model_path,map_location=device))
    model = model.to(device)

    return model

u2net = load_model(model=u2net,model_path=U2NET_MODEL_PATH, device=DEVICE)
u2netp = load_model(model=u2netp, model_path=U2NETP_MODEL_PATH, device=DEVICE)

# ------------- TRANSFORMATIONS ---------------
print("Applying Transformations ...............")
MEAN = torch.tensor([0.485, 0.456, 0.406])
STD = torch.tensor([0.229, 0.224, 0.225])

resize_shape = (320,320)

transforms = T.Compose([T.ToTensor(),
                        T.Normalize(mean=MEAN, std=STD)])

def denorm_image(image, mean, std):
    image_denorm = torch.addcmul(mean[:,None,None], image, std[:,None, None])
    image = torch.clamp(image_denorm*255., min=0., max=255.)
    image = torch.permute(image, dims=(1,2,0)).numpy().astype("uint8")

    return image

# ------------- PREPARING IMAGE BATCH ---------------
def prepare_image_batch(image_dir, resize, transforms, device):

    print("Preparing images ...............")
    image_batch = []

    for image_file in os.listdir(image_dir):
        image = Image.open(os.path.join(image_dir, image_file)).convert("RGB")
        image_resize = image.resize(resize, resample = Image.BILINEAR)

        image_trans = transforms(image_resize)
        image_batch.append(image_trans)


    image_batch = torch.stack(image_batch).to(device)

    return image_batch


# ------------- PREPARING PREDICTIONS ---------------
def prepare_predictions(model, image_batch):
    print("Starting predictions ...............")
    model.eval()

    all_results = []

    for image in image_batch:
        with torch.no_grad():
            results = model(image.unsqueeze(dim=0))

        all_results.append(torch.squeeze(results[0].cpu(), dim=(0,1)).numpy())

    return all_results


# ------------- PREPARING PREDICTIONS ---------------
def normPRED(predicted_map):
    print("Normalizing predictions ...............")
    ma = np.max(predicted_map)
    mi = np.min(predicted_map)

    map_normalize = (predicted_map - mi) / (ma-mi)

    return map_normalize


def get_prediction(og_image, result_u2net, result_u2netp):
    # Normalize the predictions
    norm_pred_u2net = normPRED(result_u2net)
    norm_pred_u2netp = normPRED(result_u2netp)
    
    # Convert to uint8
    pred_u2net = (norm_pred_u2net * 255).astype(np.uint8)
    pred_u2netp = (norm_pred_u2netp * 255).astype(np.uint8)
    
    return pred_u2net, pred_u2netp

# Create output directory if it doesn't exist
os.makedirs('subset_output_u2', exist_ok=True)

# Process each image and save predictions
image_files = os.listdir(TEST_IMAGE_DIR)
image_batch = prepare_image_batch(image_dir=TEST_IMAGE_DIR,
                                resize=resize_shape,
                                transforms=transforms,
                                device=DEVICE)

predictions_u2net = prepare_predictions(u2net, image_batch)
predictions_u2netp = prepare_predictions(u2netp, image_batch)

print("Saving predictions ...............")
for idx, image_file in enumerate(image_files):
    # Get predictions for both models
    pred_u2net, pred_u2netp = get_prediction(
        image_file,
        predictions_u2net[idx],
        predictions_u2netp[idx]
    )
    
    # Save U2NET prediction
    u2net_output = Image.fromarray(pred_u2net)
    u2net_output.save(os.path.join('subset_output_u2', f'u2net_{image_file}'))
    
    # Save U2NETP prediction
    u2netp_output = Image.fromarray(pred_u2netp)
    u2netp_output.save(os.path.join('subset_output_u2', f'u2netp_{image_file}'))

print("Predictions saved successfully!")
