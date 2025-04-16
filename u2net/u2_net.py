import os 
from PIL import Image
import numpy as np
import torch
from torchinfo import summary
import torchvision.transforms as T
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from model import U2NET, U2NETP

def load_model(model, model_path, device):
    model.load_state_dict(torch.load(model_path,map_location=device))
    model = model.to(device)
    return model

def denorm_image(image, mean, std):
    image_denorm = torch.addcmul(mean[:,None,None], image, std[:,None, None])
    image = torch.clamp(image_denorm*255., min=0., max=255.)
    image = torch.permute(image, dims=(1,2,0)).numpy().astype("uint8")
    return image

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

def prepare_predictions(model, image_batch):
    print("Starting predictions ...............")
    model.eval()
    all_results = []

    for image in image_batch:
        with torch.no_grad():
            results = model(image.unsqueeze(dim=0))
        all_results.append(torch.squeeze(results[0].cpu(), dim=(0,1)).numpy())

    return all_results

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

def apply_mask(image_path, mask):
    original = Image.open(image_path).convert('RGBA')
    mask = Image.fromarray(mask)
    mask = mask.resize(original.size, Image.BILINEAR)
    alpha = mask.copy()
    result = Image.new('RGBA', original.size, (0,0,0,0))
    result.paste(original, (0,0))
    result.putalpha(alpha)
    return result

def main():
    # Define paths
    U2NET_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'u2net.pth')
    U2NETP_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'u2netp.pth')
    TEST_IMAGE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'subset')

    # Set device
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Initialize models
    u2net = U2NET(in_ch=3,out_ch=1)
    u2netp = U2NETP(in_ch=3,out_ch=1)

    # Load models
    u2net = load_model(model=u2net, model_path=U2NET_MODEL_PATH, device=DEVICE)
    u2netp = load_model(model=u2netp, model_path=U2NETP_MODEL_PATH, device=DEVICE)

    # Define transformations
    print("Applying Transformations ...............")
    MEAN = torch.tensor([0.485, 0.456, 0.406])
    STD = torch.tensor([0.229, 0.224, 0.225])
    resize_shape = (320,320)
    transforms = T.Compose([T.ToTensor(), T.Normalize(mean=MEAN, std=STD)])

    # Create output directories
    os.makedirs('subset_output_u2', exist_ok=True)
    os.makedirs('subset_output_u2p', exist_ok=True)

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
        
        image_path = os.path.join(TEST_IMAGE_DIR, image_file)
        
        # Save U2NET prediction
        result_u2net = apply_mask(image_path, pred_u2net)
        result_u2net.save(os.path.join('subset_output_u2', f'u2net_{image_file.split(".")[0]}.png'))
        
        # Save U2NETP prediction
        u2netp_output = apply_mask(image_path, pred_u2netp)
        u2netp_output.save(os.path.join('subset_output_u2p', f'u2netp_{image_file.split(".")[0]}.png'))

    print("Predictions saved successfully!")

if __name__ == "__main__":
    main()
