import os
import time
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from omp4py import *
import sys

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the u2net module
from u2net.u2_net import remove_background_single_image
import csv
from datetime import datetime
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Benchmark U2Net models scaling')
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
OUTPUT_DIR = os.path.join(PARENT_DIR, "benchmarking", "scaling")
NUM_IMAGES = 50
REPEAT_COUNT = 3
THREAD_COUNTS = [1, 2, 4, 8, 16, 32]
MODEL_NAME = args.model

print(f"Script directory: {SCRIPT_DIR}")
print(f"Subset directory: {SUBSET_DIR}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Model name: {MODEL_NAME}")

# Check if subset directory exists
if not os.path.exists(SUBSET_DIR):
    print(f"ERROR: Subset directory {SUBSET_DIR} does not exist!")
    sys.exit(1)

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_subset_images(num_images):
    """
    Get a subset of images from the subset directory
    """
    all_images = [f for f in os.listdir(SUBSET_DIR) if f.endswith('.jpg')]
    all_images.sort()
    print(f"Found {len(all_images)} images in {SUBSET_DIR}")
    return all_images[:num_images]

@omp
def process_images_strong_scaling(image_files, num_threads):
    """
    Process images in parallel using OpenMP for strong scaling
    """
    print(f"Starting strong scaling with {len(image_files)} images and {num_threads} threads")
    
    with omp("parallel for"):
        for i in range(len(image_files)):
            image_file = image_files[i]
            image_path = os.path.join(SUBSET_DIR, image_file)
            print(f"Processing image: {image_path}")
            remove_background_single_image(image_path, output_path=None, device="cpu", model=MODEL_NAME)

def benchmark_omp_strong_scaling(num_threads):
    """
    Benchmark OpenMP processing with strong scaling (fixed problem size)
    """
    print(f"Benchmarking OpenMP strong scaling with {num_threads} threads for {MODEL_NAME}...")
    times = []
    
    omp_set_num_threads(num_threads)
    print(f"Set OpenMP threads to {num_threads}")
    
    for _ in range(REPEAT_COUNT):
        start_time = time.time()
        
        image_files = get_subset_images(NUM_IMAGES)
        process_images_strong_scaling(image_files, num_threads)
        
        end_time = time.time()
        times.append(end_time - start_time)
    
    return np.mean(times), np.std(times)

@omp
def process_images_weak_scaling(image_files, num_threads):
    """
    Process images in parallel using OpenMP for weak scaling
    """
    print(f"Starting weak scaling with {len(image_files)} images and {num_threads} threads")
    
    with omp("parallel for"):
        for i in range(len(image_files)):
            image_file = image_files[i]
            image_path = os.path.join(SUBSET_DIR, image_file)
            print(f"Processing image: {image_path}")
            remove_background_single_image(image_path, output_path=None, device="cpu", model=MODEL_NAME)

def benchmark_omp_weak_scaling(num_threads):
    """
    Benchmark OpenMP processing with weak scaling (problem size grows with threads)
    """
    print(f"Benchmarking OpenMP weak scaling with {num_threads} threads for {MODEL_NAME}...")
    times = []
    
    omp_set_num_threads(num_threads)
    print(f"Set OpenMP threads to {num_threads}")
    
    # For weak scaling, each thread processes the same number of images
    base_images = NUM_IMAGES // max(THREAD_COUNTS)  # Base number of images per thread
    total_images = base_images * num_threads
    
    print(f"Base images per thread: {base_images}")
    print(f"Total images to process: {total_images}")
    
    images_to_process = get_subset_images(total_images)
    
    for _ in range(REPEAT_COUNT):
        start_time = time.time()
        
        process_images_weak_scaling(images_to_process, num_threads)
        
        end_time = time.time()
        times.append(end_time - start_time)
    
    return np.mean(times), np.std(times), total_images

def plot_scaling_results(strong_scaling_results, weak_scaling_results):
    """
    Plot strong and weak scaling results
    """
    # Strong scaling plot
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1)
    
    threads = [result[0] for result in strong_scaling_results]
    times = [result[1] for result in strong_scaling_results]
    std_devs = [result[2] for result in strong_scaling_results]
    
    plt.errorbar(threads, times, yerr=std_devs, fmt='o-', capsize=5)
    plt.title(f'{MODEL_NAME.upper()} Strong Scaling (Fixed Problem Size: {NUM_IMAGES} images)')
    plt.xlabel('Number of Threads')
    plt.ylabel('Execution Time (seconds)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Strong scaling speedup
    plt.subplot(2, 2, 2)
    # Calculate speedup relative to single thread
    single_thread_time = next((t for th, t, _ in strong_scaling_results if th == 1), times[0])
    speedups = [single_thread_time / t for t in times]
    
    plt.plot(threads, speedups, 'o-', label='Actual Speedup')
    plt.plot(threads, threads, '--', label='Ideal Linear Speedup')
    plt.title(f'{MODEL_NAME.upper()} Strong Scaling Speedup')
    plt.xlabel('Number of Threads')
    plt.ylabel('Speedup')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Weak scaling plot
    plt.subplot(2, 2, 3)
    
    threads = [result[0] for result in weak_scaling_results]
    times = [result[1] for result in weak_scaling_results]
    std_devs = [result[2] for result in weak_scaling_results]
    total_images = [result[3] for result in weak_scaling_results]
    
    # Label each point with the number of images
    for i, (x, y, images) in enumerate(zip(threads, times, total_images)):
        plt.annotate(f"{images} imgs", 
                    (x, y), 
                    textcoords="offset points",
                    xytext=(0,10), 
                    ha='center')
    
    plt.errorbar(threads, times, yerr=std_devs, fmt='o-', capsize=5)
    plt.title(f'{MODEL_NAME.upper()} Weak Scaling\n(Images per Thread: {NUM_IMAGES // max(THREAD_COUNTS)})')
    plt.xlabel('Number of Threads')
    plt.ylabel('Execution Time (seconds)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Weak scaling efficiency
    plt.subplot(2, 2, 4)
    
    # Efficiency = T1/(Tp*p) where T1 is time with 1 thread, Tp is time with p threads
    single_thread_time = next((t for th, t, _, _ in weak_scaling_results if th == 1), times[0])
    base_imgs_per_thread = NUM_IMAGES // max(THREAD_COUNTS)
    
    # Calculate efficiency (should be 1.0 in ideal case)
    efficiencies = [single_thread_time / (times[i] * threads[i] / threads[0]) 
                   for i in range(len(threads))]
    
    plt.plot(threads, efficiencies, 'o-', label='Efficiency')
    plt.axhline(y=1.0, linestyle='--', color='r', label='Ideal Efficiency')
    plt.title(f'{MODEL_NAME.upper()} Weak Scaling Efficiency')
    plt.xlabel('Number of Threads')
    plt.ylabel('Parallel Efficiency')
    plt.ylim(0, 1.2)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{MODEL_NAME}_scaling_results.png'))
    
    # Create a second figure showing weak scaling with images on x-axis
    plt.figure(figsize=(8, 6))
    plt.errorbar(total_images, times, yerr=std_devs, fmt='o-', capsize=5)
    
    # Label each point with thread count
    for i, (x, y, thread) in enumerate(zip(total_images, times, threads)):
        plt.annotate(f"{thread} threads", 
                    (x, y), 
                    textcoords="offset points",
                    xytext=(0,10), 
                    ha='center')
    
    plt.title(f'{MODEL_NAME.upper()} Weak Scaling by Image Count')
    plt.xlabel('Total Number of Images')
    plt.ylabel('Execution Time (seconds)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{MODEL_NAME}_weak_scaling_by_images.png'))
    
    plt.close('all')

def save_scaling_results_to_csv(strong_scaling_results, weak_scaling_results):
    """
    Save scaling results to CSV files
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    strong_csv = os.path.join(OUTPUT_DIR, f'{MODEL_NAME}_strong_scaling_{timestamp}.csv')
    with open(strong_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Threads', 'Mean Time (s)', 'Std Dev (s)', 'Images', 'Repeats'])
        
        for threads, mean_time, std_time in strong_scaling_results:
            writer.writerow([threads, f"{mean_time:.4f}", f"{std_time:.4f}", NUM_IMAGES, REPEAT_COUNT])
    
    # Weak scaling CSV
    weak_csv = os.path.join(OUTPUT_DIR, f'{MODEL_NAME}_weak_scaling_{timestamp}.csv')
    with open(weak_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Threads', 'Mean Time (s)', 'Std Dev (s)', 'Images', 'Repeats'])
        
        for threads, mean_time, std_time, total_images in weak_scaling_results:
            writer.writerow([threads, f"{mean_time:.4f}", f"{std_time:.4f}", total_images, REPEAT_COUNT])
    
    print(f"Strong scaling results saved to {strong_csv}")
    print(f"Weak scaling results saved to {weak_csv}")
    return strong_csv, weak_csv

def main():
    print(f"Starting {MODEL_NAME.upper()} scaling benchmark with {NUM_IMAGES} images, {REPEAT_COUNT} repetitions")
    
    strong_scaling_results = []
    for num_threads in THREAD_COUNTS:
        mean_time, std_time = benchmark_omp_strong_scaling(num_threads)
        strong_scaling_results.append((num_threads, mean_time, std_time))
        print(f"OpenMP Strong Scaling ({num_threads} threads): {mean_time:.2f} ± {std_time:.2f} seconds")
    
    weak_scaling_results = []
    for num_threads in THREAD_COUNTS:
        mean_time, std_time, total_images = benchmark_omp_weak_scaling(num_threads)
        weak_scaling_results.append((num_threads, mean_time, std_time, total_images))
        print(f"OpenMP Weak Scaling ({num_threads} threads, {total_images} images): {mean_time:.2f} ± {std_time:.2f} seconds")
    
    plot_scaling_results(strong_scaling_results, weak_scaling_results)
    
    strong_csv, weak_csv = save_scaling_results_to_csv(strong_scaling_results, weak_scaling_results)
    
    print(f"Scaling benchmark results saved to {OUTPUT_DIR}")
    print(f"CSV files: {strong_csv}, {weak_csv}")

if __name__ == "__main__":
    main() 