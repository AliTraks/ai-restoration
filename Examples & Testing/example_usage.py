"""
Example usage of AI Image & Video Restoration System

This script demonstrates various use cases of the restoration pipeline.
"""

import cv2
import numpy as np
from pathlib import Path
from models import ModelManager
from restoration_pipeline import RestorationPipeline
from utils import (
    load_image, save_image, save_comparison,
    calculate_metrics, is_grayscale, format_time,
    create_output_directory
)

def example_1_basic_restoration():
    """Example 1: Basic image restoration with default settings"""
    print("=" * 60)
    print("Example 1: Basic Image Restoration")
    print("=" * 60)
    
    # Initialize models
    print("\n1. Loading models...")
    model_manager = ModelManager()
    model_manager.load_models()
    pipeline = RestorationPipeline(model_manager)
    
    # Load and process image
    print("\n2. Loading image...")
    image = load_image("examples/input/old_photo.jpg")
    print(f"   Input resolution: {image.shape[1]}x{image.shape[0]}")
    
    print("\n3. Processing with default settings...")
    restored, stats = pipeline.restore_image(image)
    
    print(f"\n4. Results:")
    print(f"   Output resolution: {restored.shape[1]}x{restored.shape[0]}")
    print(f"   Processing time: {format_time(stats['total_time'])}")
    print(f"   Steps performed: {', '.join(stats['steps'])}")
    
    # Save results
    output_dir = create_output_directory("examples/output")
    save_image(restored, output_dir / "restored_basic.jpg")
    save_comparison(image, restored, output_dir / "comparison_basic.jpg")
    print(f"\n5. Saved to: {output_dir}")


def example_2_grayscale_colorization():
    """Example 2: Colorize a black and white image using DDColor"""
    print("\n" + "=" * 60)
    print("Example 2: Black & White Image Colorization (DDColor)")
    print("=" * 60)
    
    model_manager = ModelManager()
    model_manager.load_models()
    pipeline = RestorationPipeline(model_manager)
    
    print("\n1. Loading grayscale image...")
    image = load_image("examples/input/bw_photo.jpg")
    
    if is_grayscale(image):
        print("   ✓ Grayscale image detected")
    
    print("\n2. Applying colorization...")
    restored, stats = pipeline.restore_image(
        image,
        apply_super_resolution=True,
        apply_colorization=True,
        sr_scale=4.0,
        colorize_render_factor=35
    )
    
    print(f"\n3. Results:")
    print(f"   Processing time: {format_time(stats['total_time'])}")
    print(f"   Colorization time: {stats['timings'].get('colorization', 0):.2f}s")
    
    output_dir = create_output_directory("examples/output")
    save_image(restored, output_dir / "colorized.jpg")
    save_comparison(image, restored, output_dir / "comparison_colorized.jpg", 
                   "Original (B&W)", "Colorized")
    print(f"\n4. Saved to: {output_dir}")


def example_3_custom_parameters():
    """Example 3: Custom enhancement parameters"""
    print("\n" + "=" * 60)
    print("Example 3: Custom Enhancement Parameters")
    print("=" * 60)
    
    model_manager = ModelManager()
    model_manager.load_models()
    pipeline = RestorationPipeline(model_manager)
    
    image = load_image("examples/input/noisy_photo.jpg")
    
    print("\n1. Testing different parameter combinations...")
    
    # Configuration 1: Light enhancement
    print("\n   Config 1: Light Enhancement")
    restored_light, stats_light = pipeline.restore_image(
        image,
        apply_super_resolution=True,
        apply_denoising=True,
        sr_scale=2.0,
        denoise_strength=5,
        detail_strength=1.2
    )
    print(f"      Time: {format_time(stats_light['total_time'])}")
    
    # Configuration 2: Aggressive enhancement
    print("\n   Config 2: Aggressive Enhancement")
    restored_aggressive, stats_aggressive = pipeline.restore_image(
        image,
        apply_super_resolution=True,
        apply_denoising=True,
        sr_scale=4.0,
        denoise_strength=20,
        detail_strength=1.8
    )
    print(f"      Time: {format_time(stats_aggressive['total_time'])}")
    
    # Save comparisons
    output_dir = create_output_directory("examples/output")
    save_comparison(restored_light, restored_aggressive, 
                   output_dir / "comparison_parameters.jpg",
                   "Light Enhancement", "Aggressive Enhancement")
    print(f"\n2. Saved comparison to: {output_dir}")


def example_4_quality_metrics():
    """Example 4: Calculate quality metrics"""
    print("\n" + "=" * 60)
    print("Example 4: Quality Metrics Analysis")
    print("=" * 60)
    
    model_manager = ModelManager()
    model_manager.load_models()
    pipeline = RestorationPipeline(model_manager)
    
    image = load_image("examples/input/degraded_photo.jpg")
    
    print("\n1. Processing image...")
    restored, stats = pipeline.restore_image(image)
    
    print("\n2. Calculating quality metrics...")
    metrics = calculate_metrics(image, restored)
    
    print("\n3. Quality Metrics:")
    print(f"   PSNR: {metrics['psnr']:.2f} dB" if metrics['psnr'] else "   PSNR: N/A")
    print(f"   MSE: {metrics['mse']:.2f}")
    print(f"   Original Sharpness: {metrics['original_sharpness']:.2f}")
    print(f"   Restored Sharpness: {metrics['restored_sharpness']:.2f}")
    
    if metrics['sharpness_improvement']:
        improvement = (metrics['sharpness_improvement'] - 1) * 100
        print(f"   Sharpness Improvement: +{improvement:.1f}%")


def example_5_batch_processing():
    """Example 5: Batch process multiple images"""
    print("\n" + "=" * 60)
    print("Example 5: Batch Image Processing")
    print("=" * 60)
    
    model_manager = ModelManager()
    model_manager.load_models()
    pipeline = RestorationPipeline(model_manager)
    
    print("\n1. Batch processing directory: examples/input/batch/")
    
    stats = pipeline.batch_restore_images(
        "examples/input/batch",
        "examples/output/batch_results",
        apply_super_resolution=True,
        apply_denoising=True,
        sr_scale=4.0
    )
    
    print(f"\n2. Batch Processing Results:")
    print(f"   Total images: {stats['total_images']}")
    print(f"   Successfully processed: {stats['processed_images']}")
    print(f"   Failed: {stats['failed_images']}")
    
    if stats['image_stats']:
        avg_time = np.mean([img['total_time'] for img in stats['image_stats']])
        print(f"   Average processing time: {format_time(avg_time)}")


def example_6_video_restoration():
    """Example 6: Video restoration"""
    print("\n" + "=" * 60)
    print("Example 6: Video Restoration")
    print("=" * 60)
    
    model_manager = ModelManager()
    model_manager.load_models()
    pipeline = RestorationPipeline(model_manager)
    
    print("\n1. Processing video (first 100 frames)...")
    print("   This may take several minutes...")
    
    def progress_callback(progress, frame_num, total_frames, frame_time):
        if frame_num % 10 == 0:  # Print every 10 frames
            print(f"   Progress: {progress*100:.1f}% ({frame_num}/{total_frames})")
    
    stats = pipeline.restore_video(
        "examples/input/old_video.mp4",
        "examples/output/restored_video.mp4",
        apply_super_resolution=True,
        apply_denoising=True,
        sr_scale=2.0,  # Lower scale for videos
        max_frames=100,  # Limit frames for demo
        progress_callback=progress_callback
    )
    
    print(f"\n2. Video Processing Results:")
    print(f"   Processed frames: {stats['processed_frames']}")
    print(f"   Input resolution: {stats['input_resolution']}")
    print(f"   Output resolution: {stats['output_resolution']}")
    print(f"   Average time per frame: {stats['average_frame_time']:.2f}s")
    print(f"   Total processing time: {format_time(stats['total_processing_time'])}")


def example_7_selective_enhancement():
    """Example 7: Selective enhancement (choose specific operations)"""
    print("\n" + "=" * 60)
    print("Example 7: Selective Enhancement")
    print("=" * 60)
    
    model_manager = ModelManager()
    model_manager.load_models()
    pipeline = RestorationPipeline(model_manager)
    
    image = load_image("examples/input/photo.jpg")
    output_dir = create_output_directory("examples/output")
    
    # Only super-resolution
    print("\n1. Super-resolution only...")
    sr_only, _ = pipeline.restore_image(
        image,
        apply_super_resolution=True,
        apply_denoising=False,
        apply_colorization=False,
        apply_detail_enhancement=False
    )
    save_image(sr_only, output_dir / "sr_only.jpg")
    
    # Only denoising
    print("2. Denoising only...")
    denoise_only, _ = pipeline.restore_image(
        image,
        apply_super_resolution=False,
        apply_denoising=True,
        apply_colorization=False,
        apply_detail_enhancement=False
    )
    save_image(denoise_only, output_dir / "denoise_only.jpg")
    
    # Only detail enhancement
    print("3. Detail enhancement only...")
    detail_only, _ = pipeline.restore_image(
        image,
        apply_super_resolution=False,
        apply_denoising=False,
        apply_colorization=False,
        apply_detail_enhancement=True
    )
    save_image(detail_only, output_dir / "detail_only.jpg")
    
    print(f"\n4. Results saved to: {output_dir}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("AI Image & Video Restoration - Example Usage")
    print("=" * 60)
    print("\nThis script demonstrates various use cases of the restoration system.")
    print("Each example will process images and save results to examples/output/")
    print("\nNote: First run will download model weights (~500MB)")
    
    # Run examples
    try:
        example_1_basic_restoration()
        # example_2_grayscale_colorization()
        # example_3_custom_parameters()
        # example_4_quality_metrics()
        # example_5_batch_processing()
        # example_6_video_restoration()
        # example_7_selective_enhancement()
        
        print("\n" + "=" * 60)
        print("Examples completed! Check examples/output/ for results.")
        print("=" * 60 + "\n")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("Please ensure example images exist in examples/input/")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()