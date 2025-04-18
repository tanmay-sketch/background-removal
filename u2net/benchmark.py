import os
import time
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

# Define transform for image preprocessing
MEAN = torch.tensor([0.485, 0.456, 0.406])
STD = torch.tensor([0.229, 0.224, 0.225])
transform = T.Compose([T.ToTensor(), T.Normalize(mean=MEAN, std=STD)])

# Parse command line arguments
parser = argparse.ArgumentParser(description='Benchmark U2Net models')
parser.add_argument(
    'model', 
    type=str, 
    choices=['u2net', 'u2netp'], 
    help='Model to benchmark (u2net or u2netp)'
)
parser.add_argument('--save-cuda-images', action='store_true', help='Save images processed by CUDA')
args = parser.parse_args()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
SUBSET_DIR = os.path.join(PARENT_DIR, "subset")
OUTPUT_DIR = os.path.join(PARENT_DIR, "benchmarking", "results")
NUM_IMAGES = 50 
REPEAT_COUNT = 5
OMP_THREADS = 10
MODEL_NAME = args.model

# Create output directory "results" if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_subset_images(num_images):
    """
    Get a subset of images from the subset directory

    Args:
        num_images (int): Number of images to return

    Returns:
        list: List of image filenames
    """
    all_images = [f for f in os.listdir(SUBSET_DIR) if f.endswith('.jpg')]
    all_images.sort()
    return all_images[:num_images]

def benchmark_cpu():
    """
    CPU Benchmark

    Args:
        None

    Returns:
        tuple: Tuple containing the mean and standard deviation of the benchmark times
    """
    print(f"Benchmarking CPU processing for {MODEL_NAME}...")
    times = []
    
    for run in range(REPEAT_COUNT):
        print(f"    CPU run {run+1}/{REPEAT_COUNT}...")
        start_time = time.time()
        
        for image_file in get_subset_images(NUM_IMAGES):
            image_path = os.path.join(SUBSET_DIR, image_file)
            remove_background_single_image(image_path, output_path=None, device="cpu", model=MODEL_NAME)
        
        end_time = time.time()
        run_time = end_time - start_time
        times.append(run_time)
        print(f"    CPU run {run+1} completed in {run_time:.2f} seconds")
    
    mean_time = np.mean(times)
    std_time = np.std(times)
    print(f"    CPU benchmark completed. Average time: {mean_time:.2f} ± {std_time:.2f} seconds")
    return mean_time, std_time

@omp
def process_images_omp(image_files):
    """
    Process images in parallel using OpenMP

    Args:
        image_files (list): List of image filenames

    Returns:
        None
    """
    with omp("parallel for"):
        for i in range(len(image_files)):
            image_file = image_files[i]
            image_path = os.path.join(SUBSET_DIR, image_file)
            remove_background_single_image(image_path, output_path=None, device="cpu", model=MODEL_NAME)

def benchmark_omp():
    """
    OpenMP Benchmark with 10 threads

    Args:
        None

    Returns:
        tuple: Tuple containing the mean and standard deviation of the benchmark times
    """
    print(f"Benchmarking OpenMP processing with {OMP_THREADS} threads for {MODEL_NAME}...")
    times = []
    
    omp_set_num_threads(OMP_THREADS)
    
    for run in range(REPEAT_COUNT):
        print(f"    OpenMP run {run+1}/{REPEAT_COUNT}...")
        
        image_files = get_subset_images(NUM_IMAGES)
        start_time = time.time()
        process_images_omp(image_files)
        
        end_time = time.time()
        run_time = end_time - start_time
        times.append(run_time)
        print(f"  OpenMP run {run+1} completed in {run_time:.2f} seconds")
    
    mean_time = np.mean(times)
    std_time = np.std(times)
    print(f"OpenMP benchmark completed. Average time: {mean_time:.2f} ± {std_time:.2f} seconds")
    return mean_time, std_time

def benchmark_cuda(model_name, num_runs=10, save_images=False):
    """
    Benchmark CUDA implementation

    Args: 
        model_name (str): Name of the model to benchmark
        num_runs (int): Number of runs to perform
        save_images (bool): Whether to save images

    Returns:
        tuple: Tuple containing the mean and standard deviation of the benchmark times
    """

    print(f"\nBenchmarking CUDA implementation for {model_name}...")
    # print(f"    save_images flag: {save_images}")  # Debug print
    
    if not torch.cuda.is_available():
        print("CUDA is not available. Skipping CUDA benchmark.")
        return None, None
    
    # Initialize model
    if model_name == "u2net":
        net = U2NET(in_ch=3, out_ch=1)
    else:  # u2netp
        net = U2NETP(in_ch=3, out_ch=1)
        
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'u2net', 'model', f'{model_name}.pth')
    print(f"  Loading model from: {model_path}")
    u2net = load_model(model=net, model_path=model_path, device="cuda")
    u2net.eval()
    
    image_files = get_subset_images(NUM_IMAGES)
    if not image_files:
        print("No images found for CUDA benchmarking.")
        return None, None
    
    print(f"  Found {len(image_files)} images for benchmarking")
    
    # Comment out image saving setup
    """
    # Force save_images to True if the flag is set
    save_images = args.save_cuda_images
    
    # Create output directory for CUDA images in a fixed location
    if save_images:
        # Create a directory in the parent directory, making it easier to find
        cuda_output_dir = os.path.join(PARENT_DIR, "cuda_results")
        print(f"    Image saving enabled. Output directory: {cuda_output_dir}")
        
        try:
            os.makedirs(cuda_output_dir, exist_ok=True)
            print(f"    Created output directory: {cuda_output_dir}")
            
            # Test if we can write to the directory
            test_file = os.path.join(cuda_output_dir, "test_write.txt")
            with open(test_file, 'w') as f:
                f.write("Test write permission")
            os.remove(test_file)
            print(f"    Successfully tested write permissions to directory")
            
        except Exception as e:
            print(f"    ERROR with output directory {cuda_output_dir}: {e}")
            print("    *** Image saving will be disabled due to directory error ***")
            save_images = False
    else:
        print("    Image saving is disabled")
    """
    # Always disable image saving for benchmarks
    save_images = False
    
    # Prepare all images for batch processing
    print("  Preparing images for batch processing...")
    image_batch = []
    original_images = []  # Store original images for mask application
    
    for image_file in image_files:
        image_path = os.path.join(SUBSET_DIR, image_file)
        image = Image.open(image_path).convert('RGB')
        original_images.append(image)  # Store original image
        image = image.resize((320, 320), Image.BILINEAR)
        image_tensor = transform(image)
        image_batch.append(image_tensor)
    
    # Stack all images into a single batch tensor
    batch_tensor = torch.stack(image_batch).to("cuda")
    print(f"  Batch tensor shape: {batch_tensor.shape}")
    
    # Warm-up run
    print("  Performing warm-up run...")
    with torch.no_grad():
        warmup_results = u2net(batch_tensor)
        _ = warmup_results[0]
    torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for run in range(num_runs):
        print(f"  CUDA run {run+1}/{num_runs}...")
        start_time = time.time()
        
        # Process all images in a single batch
        with torch.no_grad():
            results = u2net(batch_tensor)
            predictions = results[0]
        torch.cuda.synchronize()
        
        # Process each prediction (but don't save)
        for i, prediction in enumerate(predictions):
            # Extract and process the prediction
            pred = torch.squeeze(prediction.cpu(), dim=(0,1)).numpy()
            pred = normPRED(pred)
            pred = (pred * 255).astype(np.uint8)
            
            """
            # Comment out image saving code
            # Only process for saving in the first run
            if save_images and run == 0:
                print(f"    Processing image {i+1}/{len(predictions)}: {image_files[i]}")
                
                # Create the output filename with png extension
                output_filename = f"cuda_{model_name}_{image_files[i]}"
                if not output_filename.lower().endswith(('.png', '.jpg')):
                    output_filename = output_filename.split('.')[0] + '.png'
                elif output_filename.lower().endswith('.jpg'):
                    output_filename = output_filename.replace('.jpg', '.png')
                
                output_path = os.path.join(cuda_output_dir, output_filename)
                
                try:
                    # Create a mask image and resize it to match the original image
                    mask = Image.fromarray(pred)
                    mask = mask.resize(original_images[i].size, Image.BILINEAR)
                    
                    # Apply the mask directly to create a background-removed image
                    # Method 1: Create an RGBA image
                    original_array = np.array(original_images[i])
                    mask_array = np.array(mask)
                    
                    # Create an RGBA image (RGB + Alpha channel)
                    rgba = np.zeros((original_array.shape[0], original_array.shape[1], 4), dtype=np.uint8)
                    rgba[:,:,0:3] = original_array
                    rgba[:,:,3] = mask_array  # Alpha channel
                    
                    # Save the RGBA image
                    Image.fromarray(rgba, mode='RGBA').save(output_path, format='PNG')
                    
                    if os.path.exists(output_path):
                        print(f"    ✓ Successfully saved image to: {output_path}")
                    else:
                        print(f"    ✗ Failed to save image: {output_path} (file not found after save)")
                
                except Exception as e:
                    print(f"    ✗ ERROR saving image {i+1}: {str(e)}")
            """
        
        end_time = time.time()
        run_time = end_time - start_time
        times.append(run_time)
        print(f"  CUDA run {run+1} completed in {run_time:.4f} seconds")
    
    """
    # Comment out summary section
    # Print summary of saved images
    if save_images:
        try:
            saved_files = os.listdir(cuda_output_dir)
            print(f"\n=== CUDA Image Saving Summary ===")
            print(f"Output directory: {cuda_output_dir}")
            print(f"Number of images saved: {len([f for f in saved_files if f.startswith('cuda_')])}")
            print(f"All files in directory: {saved_files}")
            print("==============================\n")
        except Exception as e:
            print(f"Error listing output directory: {e}")
    """
    
    avg_time = sum(times) / len(times)
    std_time = np.std(times)
    print(f"CUDA benchmark completed. Average time: {avg_time:.4f} ± {std_time:.4f} seconds")
    return avg_time, std_time

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
    
    # Use fixed filename without timestamp
    output_path = os.path.join(PARENT_DIR, "benchmarking", "results", f'{MODEL_NAME}_benchmark_results.png')
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    plt.close()

def save_results_to_csv(results):
    """
    Save benchmark results to a CSV file
    """
    # Use fixed filename without timestamp
    csv_filename = os.path.join(PARENT_DIR, "benchmarking", "results", f'{MODEL_NAME}_benchmark_results.csv')
    
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
    
    # Ensure output directory exists
    results_dir = os.path.join(PARENT_DIR, "benchmarking", "results")
    os.makedirs(results_dir, exist_ok=True)
    
    results = {}
    
    print("\n" + "="*50)
    print("RUNNING CPU BENCHMARK")
    print("="*50)
    cpu_mean, cpu_std = benchmark_cpu()
    results['CPU'] = (cpu_mean, cpu_std)
    print(f"CPU: {cpu_mean:.2f} ± {cpu_std:.2f} seconds")
    
    print("\n" + "="*50)
    print("RUNNING OPENMP BENCHMARK")
    print("="*50)
    omp_mean, omp_std = benchmark_omp()
    results['OpenMP'] = (omp_mean, omp_std)
    print(f"OpenMP ({OMP_THREADS} threads): {omp_mean:.2f} ± {omp_std:.2f} seconds")
    
    print("\n" + "="*50)
    print("RUNNING CUDA BENCHMARK")
    print("="*50)
    cuda_mean, cuda_std = benchmark_cuda(MODEL_NAME, num_runs=REPEAT_COUNT, save_images=args.save_cuda_images)
    if cuda_mean is not None:
        results['CUDA'] = (cuda_mean, cuda_std)
        print(f"CUDA: {cuda_mean:.2f} ± {cuda_std:.2f} seconds")
    
    print("\n" + "="*50)
    print("BENCHMARK SUMMARY")
    print("="*50)
    for method, (mean_time, std_time) in results.items():
        print(f"{method}: {mean_time:.2f} ± {std_time:.2f} seconds")
    
    plot_results(results)
    
    csv_file = save_results_to_csv(results)
    
    print(f"\nBenchmark results saved to {results_dir}")
    print(f"Output structure is now:")
    print(f"- {os.path.join(results_dir, f'{MODEL_NAME}_benchmark_results.csv')}")
    print(f"- {os.path.join(results_dir, f'{MODEL_NAME}_benchmark_results.png')}")

if __name__ == "__main__":
    main()
