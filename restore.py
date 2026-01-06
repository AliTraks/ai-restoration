#!/usr/bin/env python3
"""
Command-line interface for AI Image & Video Restoration
"""

import argparse
import sys
import cv2
from pathlib import Path
from models import ModelManager
from restoration_pipeline import RestorationPipeline
from utils import (
    load_image, save_image, save_comparison, 
    format_time, is_grayscale, get_video_info
)

def main():
    parser = argparse.ArgumentParser(
        description='AI-powered image and video restoration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Restore single image with default settings
  python restore.py --input photo.jpg --output restored.jpg
  
  # Restore with custom super-resolution scale
  python restore.py --input photo.jpg --output restored.jpg --sr-scale 4.0
  
  # Colorize black and white image
  python restore.py --input bw_photo.jpg --output color.jpg --colorize
  
  # Restore video (first 300 frames for testing)
  python restore.py --input video.mp4 --output restored.mp4 --video --max-frames 300
  
  # Batch process directory
  python restore.py --input ./input_dir --output ./output_dir --batch
  
  # Full pipeline with all enhancements
  python restore.py --input photo.jpg --output result.jpg --sr --denoise --detail --comparison
        """
    )
    
    # Input/Output
    parser.add_argument('--input', '-i', required=True, 
                       help='Input image, video, or directory path')
    parser.add_argument('--output', '-o', required=True,
                       help='Output path for restored media')
    
    # Processing mode
    parser.add_argument('--video', action='store_true',
                       help='Process as video')
    parser.add_argument('--batch', action='store_true',
                       help='Batch process directory of images')
    
    # Enhancement options
    parser.add_argument('--sr', '--super-resolution', action='store_true', default=True,
                       dest='super_resolution', help='Enable super-resolution (default: True)')
    parser.add_argument('--no-sr', action='store_false', dest='super_resolution',
                       help='Disable super-resolution')
    parser.add_argument('--denoise', action='store_true', default=True,
                       help='Enable denoising (default: True)')
    parser.add_argument('--no-denoise', action='store_false', dest='denoise',
                       help='Disable denoising')
    parser.add_argument('--colorize', action='store_true', default=False,
                       help='Enable colorization for grayscale images')
    parser.add_argument('--detail', '--detail-enhancement', action='store_true', default=True,
                       dest='detail_enhancement', help='Enable detail enhancement (default: True)')
    parser.add_argument('--no-detail', action='store_false', dest='detail_enhancement',
                       help='Disable detail enhancement')
    
    # Parameters
    parser.add_argument('--sr-scale', type=float, default=4.0,
                       help='Super-resolution scale factor (2.0-4.0, default: 4.0)')
    parser.add_argument('--denoise-strength', type=int, default=10,
                       help='Denoising strength (1-30, default: 10)')
    parser.add_argument('--colorize-quality', type=int, default=35,
                       help='Colorization quality (10-40, default: 35)')
    parser.add_argument('--detail-strength', type=float, default=1.5,
                       help='Detail enhancement strength (1.0-2.0, default: 1.5)')
    
    # Video options
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Maximum frames to process for video (default: all)')
    
    # Output options
    parser.add_argument('--comparison', action='store_true',
                       help='Save side-by-side comparison image')
    parser.add_argument('--stats', action='store_true',
                       help='Print detailed processing statistics')
    
    args = parser.parse_args()
    
    # Validate paths
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Error: Input path '{args.input}' does not exist")
        sys.exit(1)
    
    # Initialize models
    print("🚀 Loading AI models...")
    model_manager = ModelManager()
    models_loaded = model_manager.load_models(
        load_sr=args.super_resolution,
        load_colorize=args.colorize
    )
    
    if args.super_resolution and not models_loaded.get('super_resolution'):
        print("⚠️  Warning: Super-resolution model failed to load")
    if args.colorize and not models_loaded.get('colorization'):
        print("⚠️  Warning: Colorization model failed to load")
    
    pipeline = RestorationPipeline(model_manager)
    print("✅ Models loaded successfully\n")
    
    # Process based on mode
    if args.video:
        # Video processing
        print(f"🎬 Processing video: {args.input}")
        
        video_info = get_video_info(args.input)
        print(f"   Resolution: {video_info['width']}x{video_info['height']}")
        print(f"   FPS: {video_info['fps']}")
        print(f"   Frames: {video_info['frame_count']}")
        
        if args.max_frames:
            print(f"   Processing: {args.max_frames} frames")
        
        def progress_callback(progress, frame_num, total_frames, frame_time):
            print(f"\r   Progress: {progress*100:.1f}% ({frame_num}/{total_frames}) - "
                  f"{frame_time:.2f}s/frame", end='', flush=True)
        
        stats = pipeline.restore_video(
            args.input,
            args.output,
            apply_super_resolution=args.super_resolution,
            apply_denoising=args.denoise,
            apply_colorization=args.colorize,
            apply_detail_enhancement=args.detail_enhancement,
            sr_scale=args.sr_scale,
            denoise_strength=args.denoise_strength,
            colorize_render_factor=args.colorize_quality,
            detail_strength=args.detail_strength,
            max_frames=args.max_frames,
            progress_callback=progress_callback
        )
        
        print(f"\n\n✅ Video processing complete!")
        print(f"   Output: {args.output}")
        print(f"   Processed frames: {stats['processed_frames']}")
        print(f"   Average time per frame: {stats['average_frame_time']:.2f}s")
        print(f"   Total processing time: {format_time(stats['total_processing_time'])}")
        
    elif args.batch:
        # Batch processing
        print(f"📁 Batch processing directory: {args.input}")
        
        stats = pipeline.batch_restore_images(
            args.input,
            args.output,
            apply_super_resolution=args.super_resolution,
            apply_denoising=args.denoise,
            apply_colorization=args.colorize,
            apply_detail_enhancement=args.detail_enhancement,
            sr_scale=args.sr_scale,
            denoise_strength=args.denoise_strength,
            colorize_render_factor=args.colorize_quality,
            detail_strength=args.detail_strength
        )
        
        print(f"\n✅ Batch processing complete!")
        print(f"   Total images: {stats['total_images']}")
        print(f"   Successfully processed: {stats['processed_images']}")
        print(f"   Failed: {stats['failed_images']}")
        print(f"   Output directory: {args.output}")
        
    else:
        # Single image processing
        print(f"📷 Processing image: {args.input}")
        
        image = load_image(args.input)
        print(f"   Input resolution: {image.shape[1]}x{image.shape[0]}")
        
        is_gray = is_grayscale(image)
        if is_gray:
            print("   Detected: Grayscale image")
            if args.colorize:
                print("   Colorization: Enabled")
        
        print(f"\n⚙️  Enhancement settings:")
        print(f"   Super-resolution: {'Yes' if args.super_resolution else 'No'} "
              f"(scale: {args.sr_scale}x)" if args.super_resolution else "")
        print(f"   Denoising: {'Yes' if args.denoise else 'No'} "
              f"(strength: {args.denoise_strength})" if args.denoise else "")
        print(f"   Colorization: {'Yes' if args.colorize and is_gray else 'No'}")
        print(f"   Detail enhancement: {'Yes' if args.detail_enhancement else 'No'} "
              f"(strength: {args.detail_strength})" if args.detail_enhancement else "")
        
        print("\n🔄 Processing...")
        restored, stats = pipeline.restore_image(
            image,
            apply_super_resolution=args.super_resolution,
            apply_denoising=args.denoise,
            apply_colorization=args.colorize and is_gray,
            apply_detail_enhancement=args.detail_enhancement,
            sr_scale=args.sr_scale,
            denoise_strength=args.denoise_strength,
            colorize_render_factor=args.colorize_quality,
            detail_strength=args.detail_strength
        )
        
        # Save output
        save_image(restored, args.output)
        
        print(f"\n✅ Image restoration complete!")
        print(f"   Output: {args.output}")
        print(f"   Output resolution: {restored.shape[1]}x{restored.shape[0]}")
        print(f"   Processing time: {format_time(stats['total_time'])}")
        
        if args.stats:
            print(f"\n📊 Detailed statistics:")
            print(f"   Steps performed: {', '.join(stats['steps'])}")
            for step, time in stats['timings'].items():
                print(f"   {step}: {time:.3f}s")
        
        # Save comparison if requested
        if args.comparison:
            comparison_path = str(Path(args.output).parent / f"comparison_{Path(args.output).name}")
            save_comparison(image, restored, comparison_path)
            print(f"   Comparison saved: {comparison_path}")
    
    print("\n🎉 Done!")

if __name__ == '__main__':
    main()