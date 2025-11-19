# benchmark_comparison.py

import time
import numpy as np
import cv2
from dino_wrapper import run_dino
from sam_wrapper import run_sam, predictor  # reuse the loaded SAM model

from segment_anything import SamAutomaticMaskGenerator
import torch

def benchmark_sam_with_dino(image_path, confidence_threshold=0.3, num_runs=5):
    """
    Benchmark SAM with DINO providing bounding boxes
    """
    print("\n" + "="*60)
    print("BENCHMARKING: SAM with DINO (Hybrid Pipeline)")
    print("="*60)
    
    times = {"dino": [], "sam": [], "total": []}
    
    for run in range(num_runs):
        print(f"\nRun {run + 1}/{num_runs}")
        
        # Time DINO
        start_dino = time.time()
        boxes, scores, labels = run_dino(image_path, visualize=False)
        confident_indices = scores >= confidence_threshold
        filtered_boxes = boxes[confident_indices]
        dino_time = time.time() - start_dino
        print(f"  DINO: {dino_time:.3f}s ({len(filtered_boxes)} boxes)")
        
        # Time SAM with filtered boxes
        start_sam = time.time()
        masks = run_sam(image_path, filtered_boxes)
        sam_time = time.time() - start_sam
        print(f"  SAM: {sam_time:.3f}s ({len(masks)} masks)")
        
        total_time = dino_time + sam_time
        print(f"  Total: {total_time:.3f}s")
        
        times["dino"].append(dino_time)
        times["sam"].append(sam_time)
        times["total"].append(total_time)
    
    avg_dino = np.mean(times["dino"])
    avg_sam = np.mean(times["sam"])
    avg_total = np.mean(times["total"])
    
    print(f"\n{'─'*60}")
    print(f"Average over {num_runs} runs:")
    print(f"  DINO:  {avg_dino:.3f}s ± {np.std(times['dino']):.3f}s")
    print(f"  SAM:   {avg_sam:.3f}s ± {np.std(times['sam']):.3f}s")
    print(f"  Total: {avg_total:.3f}s ± {np.std(times['total']):.3f}s")
    
    return times


def benchmark_sam_alone(image_path, num_runs=5):
    """
    Benchmark SAM running on the full image without DINO.
    Uses the already loaded SAM model from sam_wrapper.
    """
    print("\n" + "="*60)
    print("BENCHMARKING: SAM Alone (Full Image Segmentation)")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Use the same predictor.model as loaded in sam_wrapper
    mask_generator = SamAutomaticMaskGenerator(
        model=predictor.model,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,
    )
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    times = []
    
    for run in range(num_runs):
        print(f"\nRun {run + 1}/{num_runs}")
        start = time.time()
        masks = mask_generator.generate(image_rgb)
        elapsed = time.time() - start
        print(f"  SAM: {elapsed:.3f}s ({len(masks)} masks generated)")
        times.append(elapsed)
    
    avg_time = np.mean(times)
    print(f"\n{'─'*60}")
    print(f"Average over {num_runs} runs:")
    print(f"  SAM Alone: {avg_time:.3f}s ± {np.std(times):.3f}s")
    
    return times


def compare_approaches(image_path, confidence_threshold=0.3, num_runs=3):
    """
    Run both benchmarks and compare results
    """
    print("\n" + "="*60)
    print(f"COMPARING APPROACHES ON: {image_path}")
    print("="*60)
    
    hybrid_times = benchmark_sam_with_dino(image_path, confidence_threshold, num_runs)
    sam_alone_times = benchmark_sam_alone(image_path, num_runs)
    
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    hybrid_avg = np.mean(hybrid_times["total"])
    sam_alone_avg = np.mean(sam_alone_times)
    speedup = sam_alone_avg / hybrid_avg
    time_saved = sam_alone_avg - hybrid_avg
    
    print(f"\nHybrid (DINO + SAM):  {hybrid_avg:.3f}s")
    print(f"SAM Alone:            {sam_alone_avg:.3f}s")
    print(f"\nSpeedup:              {speedup:.2f}x faster")
    print(f"Time Saved:           {time_saved:.3f}s ({(time_saved/sam_alone_avg)*100:.1f}% reduction)")
    
    print("\n" + "="*60)
    print("BREAKDOWN (Hybrid Approach):")
    print("="*60)
    dino_avg = np.mean(hybrid_times["dino"])
    sam_avg = np.mean(hybrid_times["sam"])
    print(f"DINO time:   {dino_avg:.3f}s ({(dino_avg/hybrid_avg)*100:.1f}% of total)")
    print(f"SAM time:    {sam_avg:.3f}s ({(sam_avg/hybrid_avg)*100:.1f}% of total)")
    
    return {
        "hybrid": hybrid_times,
        "sam_alone": sam_alone_times,
        "speedup": speedup,
        "time_saved": time_saved
    }


if __name__ == "__main__":
    image_path = "data/sample_images/test.jpg"
    results = compare_approaches(image_path, confidence_threshold=0.3, num_runs=3)
    
    print("\n✅ Benchmark complete!")
    print(f"\nConclusion: The hybrid approach is {results['speedup']:.2f}x faster than running SAM alone.")
