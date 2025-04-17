import os
import time
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the u2net module
from u2net.u2_net import remove_background_single_image
from u2net.u2_net import load_model
from u2net.u2_net import normPRED
from u2net.u2_net import apply_mask
from omp4py import *
import csv
from datetime import datetime
import torchvision.transforms as T
import argparse
from u2net.model import U2NET, U2NETP

# Parse command line arguments
parser = argparse.ArgumentParser(description='Benchmark U2Net models')
parser.add_argument(
    'model', 
    type=str, 
    choices=['u2net', 'u2netp'], 
    help='Model to benchmark (u2net or u2netp)'
)
args = parser.parse_args()

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
SUBSET_DIR = os.path.join(PARENT_DIR, "subset")
OUTPUT_DIR = os.path.join(PARENT_DIR, "benchmarking", "results")
NUM_IMAGES = 50 
REPEAT_COUNT = 5
OMP_THREADS = 10
MODEL_NAME = args.model

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_subset_images(num_images):
    """
    Get a subset of images from the subset directory
    """
    all_images = [f for f in os.listdir(SUBSET_DIR) if f.endswith('.jpg')]
    all_images.sort()
    return all_images[:num_images]

def benchmark_cpu():
    """
    CPU Benchmark
    """
    print(f"Benchmarking CPU processing for {MODEL_NAME}...")
    times = []
    
    for _ in range(REPEAT_COUNT):
        start_time = time.time()
        
        for image_file in get_subset_images(NUM_IMAGES):
            image_path = os.path.join(SUBSET_DIR, image_file)
            remove_background_single_image(image_path, output_path=None, device="cpu", model=MODEL_NAME)
        
        end_time = time.time()
        times.append(end_time - start_time)
    
    return np.mean(times), np.std(times)

@omp
def process_images_omp(image_files, output_prefix):
    """
    Process images in parallel using OpenMP
    """
    with omp("parallel for"):
        for image_file in image_files:
            image_path = os.path.join(SUBSET_DIR, image_file)
            remove_background_single_image(image_path, output_path=None, device="cpu", model=MODEL_NAME)

def benchmark_omp():
    """
    OpenMP Benchmark with 10 threads
    """
    print(f"Benchmarking OpenMP processing with {OMP_THREADS} threads for {MODEL_NAME}...")
    times = []
    
    omp4py.set_num_threads(OMP_THREADS)
    
    for _ in range(REPEAT_COUNT):
        start_time = time.time()
        
        image_files = get_subset_images(NUM_IMAGES)
        process_images_omp(image_files, "omp")
        
        end_time = time.time()
        times.append(end_time - start_time)
    
    return np.mean(times), np.std(times)

def benchmark_cuda():
    """
    CUDA Benchmark
    """
    print(f"Benchmarking CUDA processing for {MODEL_NAME}...")
    times = []
    
    if not torch.cuda.is_available():
        print("CUDA is not available. Skipping CUDA benchmark.")
        return None, None
    
    # Initialize model once
    if MODEL_NAME == "u2net":
        net = U2NET(in_ch=3, out_ch=1)
    else:  # u2netp
        net = U2NETP(in_ch=3, out_ch=1)
        
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'u2net', 'model', f'{MODEL_NAME}.pth')
    u2net = load_model(model=net, model_path=model_path, device="cuda")
    u2net.eval()
    
    # Define transformations
    MEAN = torch.tensor([0.485, 0.456, 0.406])
    STD = torch.tensor([0.229, 0.224, 0.225])
    resize_shape = (320, 320)
    transforms = T.Compose([T.ToTensor(), T.Normalize(mean=MEAN, std=STD)])
    
    for _ in range(REPEAT_COUNT):
        start_time = time.time()
        
        # Process all images in a batch
        image_files = get_subset_images(NUM_IMAGES)
        image_batch = []
        
        # Prepare all images
        for image_file in image_files:
            image_path = os.path.join(SUBSET_DIR, image_file)
            image = Image.open(image_path).convert("RGB")
            image_resize = image.resize(resize_shape, resample=Image.BILINEAR)
            image_trans = transforms(image_resize)
            image_batch.append(image_trans)
        
        # Stack all images into a single batch tensor
        batch_tensor = torch.stack(image_batch).to("cuda")
        
        # Process the entire batch at once
        with torch.no_grad():
            results = u2net(batch_tensor)
        
        # Process each result (but don't save images)
        for i, result in enumerate(results):
            pred = torch.squeeze(result.cpu(), dim=(0,1)).numpy()
            pred = normPRED(pred)
            pred = (pred * 255).astype(np.uint8)
            # Apply mask to original image but don't save
            apply_mask(os.path.join(SUBSET_DIR, image_files[i]), pred)
        
        end_time = time.time()
        times.append(end_time - start_time)
    
    return np.mean(times), np.std(times)

def plot_results(results):
    """
    Plot benchmark results
    """
    methods = list(results.keys())
    mean_times = [results[method][0] for method in methods]
    std_times = [results[method][1] for method in methods]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, mean_times, yerr=std_times, capsize=10)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}s', ha='center', va='bottom')
    
    plt.title(f'{MODEL_NAME.upper()} Background Removal Benchmark ({NUM_IMAGES} images)')
    plt.ylabel('Time (seconds)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.savefig(os.path.join(OUTPUT_DIR, f'{MODEL_NAME}_benchmark_results.png'))
    plt.close()

def save_results_to_csv(results):
    """
    Save benchmark results to a CSV file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = os.path.join(OUTPUT_DIR, f'{MODEL_NAME}_benchmark_results_{timestamp}.csv')
    
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Method', 'Mean Time (s)', 'Std Dev (s)', 'Images', 'Repeats', 'Threads'])
        
        for method, (mean_time, std_time) in results.items():
            threads = OMP_THREADS if method == 'OpenMP' else 'N/A'
            writer.writerow([method, f"{mean_time:.4f}", f"{std_time:.4f}", NUM_IMAGES, REPEAT_COUNT, threads])
    
    print(f"Results saved to {csv_filename}")
    return csv_filename

def main():
    print(f"Starting {MODEL_NAME.upper()} benchmark with {NUM_IMAGES} images, {REPEAT_COUNT} repetitions")
    
    results = {}
    
    cpu_mean, cpu_std = benchmark_cpu()
    results['CPU'] = (cpu_mean, cpu_std)
    print(f"CPU: {cpu_mean:.2f} ± {cpu_std:.2f} seconds")
    
    omp_mean, omp_std = benchmark_omp()
    results['OpenMP'] = (omp_mean, omp_std)
    print(f"OpenMP ({OMP_THREADS} threads): {omp_mean:.2f} ± {omp_std:.2f} seconds")
    
    cuda_mean, cuda_std = benchmark_cuda()
    if cuda_mean is not None:
        results['CUDA'] = (cuda_mean, cuda_std)
        print(f"CUDA: {cuda_mean:.2f} ± {cuda_std:.2f} seconds")
    
    plot_results(results)
    
    csv_file = save_results_to_csv(results)
    
    print(f"Benchmark results saved to {OUTPUT_DIR}")
    print(f"CSV file: {csv_file}")

if __name__ == "__main__":
    main()
