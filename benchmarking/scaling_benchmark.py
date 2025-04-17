import os
import time
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from u2net.u2_net import remove_background_single_image
from omp4py import *
import csv
from datetime import datetime
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Benchmark U2Net models scaling')
parser.add_argument('model', type=str, choices=['u2net', 'u2netp'], 
                    help='Model to benchmark (u2net or u2netp)')
args = parser.parse_args()

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SUBSET_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "subset")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "scaling")
NUM_IMAGES = 50
REPEAT_COUNT = 3
THREAD_COUNTS = [1, 2, 4, 8, 16, 32]
MODEL_NAME = args.model

os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_subset_images(num_images):
    """
    Get a subset of images from the subset directory
    """
    all_images = [f for f in os.listdir(SUBSET_DIR) if f.endswith('.jpg')]
    all_images.sort()
    return all_images[:num_images]

@omp
def process_images_strong_scaling(image_files, num_threads):
    """
    Process images in parallel using OpenMP for strong scaling
    """
    with omp("parallel for"):
        for image_file in image_files:
            image_path = os.path.join(SUBSET_DIR, image_file)
            remove_background_single_image(image_path, output_path=None, device="cpu", model=MODEL_NAME)

def benchmark_omp_strong_scaling(num_threads):
    """
    Benchmark OpenMP processing with strong scaling (fixed problem size)
    """
    print(f"Benchmarking OpenMP strong scaling with {num_threads} threads for {MODEL_NAME}...")
    times = []
    
    omp4py.set_num_threads(num_threads)
    
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
    with omp("parallel for"):
        for image_file in image_files:
            image_path = os.path.join(SUBSET_DIR, image_file)
            remove_background_single_image(image_path, output_path=None, device="cpu", model=MODEL_NAME)

def benchmark_omp_weak_scaling(num_threads):
    """
    Benchmark OpenMP processing with weak scaling (problem size grows with threads)
    """
    print(f"Benchmarking OpenMP weak scaling with {num_threads} threads for {MODEL_NAME}...")
    times = []
    
    # Set number of threads for OpenMP
    omp4py.set_num_threads(num_threads)
    
    # For weak scaling, each thread processes the same number of images
    base_images = NUM_IMAGES // max(THREAD_COUNTS)  # Base number of images per thread
    total_images = base_images * num_threads
    
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
    plt.figure(figsize=(12, 5))
    
    # Strong scaling subplot
    plt.subplot(1, 2, 1)
    threads = [result[0] for result in strong_scaling_results]
    times = [result[1] for result in strong_scaling_results]
    std_devs = [result[2] for result in strong_scaling_results]
    
    plt.errorbar(threads, times, yerr=std_devs, fmt='o-', capsize=5)
    plt.title(f'{MODEL_NAME.upper()} Strong Scaling')
    plt.xlabel('Number of Threads')
    plt.ylabel('Time (seconds)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Weak scaling subplot
    plt.subplot(1, 2, 2)
    threads = [result[0] for result in weak_scaling_results]
    times = [result[1] for result in weak_scaling_results]
    std_devs = [result[2] for result in weak_scaling_results]
    
    plt.errorbar(threads, times, yerr=std_devs, fmt='o-', capsize=5)
    plt.title(f'{MODEL_NAME.upper()} Weak Scaling')
    plt.xlabel('Number of Threads')
    plt.ylabel('Time (seconds)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{MODEL_NAME}_scaling_results.png'))
    plt.close()

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