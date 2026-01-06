import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict
import time
from models import ModelManager

class RestorationPipeline:
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.processing_stats = {}
        
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Prepare image for processing"""
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image
    
    def restore_image(
        self,
        image: np.ndarray,
        apply_super_resolution: bool = True,
        apply_denoising: bool = True,
        apply_colorization: bool = False,
        apply_detail_enhancement: bool = True,
        sr_scale: float = 4.0,
        denoise_strength: int = 10,
        colorize_render_factor: int = 35,
        detail_strength: float = 1.5
    ) -> Tuple[np.ndarray, Dict]:
        """
        Main restoration pipeline
        
        Args:
            image: Input image as numpy array
            apply_super_resolution: Enable super-resolution
            apply_denoising: Enable denoising
            apply_colorization: Enable colorization
            apply_detail_enhancement: Enable detail enhancement
            sr_scale: Super-resolution scale factor
            denoise_strength: Denoising strength (1-30)
            colorize_render_factor: Colorization quality (10-40)
            detail_strength: Detail enhancement strength (1.0-2.0)
            
        Returns:
            Tuple of (restored_image, processing_stats)
        """
        stats = {
            'steps': [],
            'timings': {},
            'input_shape': image.shape,
            'output_shape': None
        }
        
        result = self.preprocess_image(image.copy())
        
        # Step 1: Denoising (before upscaling to reduce computation)
        if apply_denoising and self.model_manager.denoise_model:
            start = time.time()
            result = self.model_manager.denoise_model.denoise(result, strength=denoise_strength)
            stats['steps'].append('Denoising')
            stats['timings']['denoising'] = time.time() - start
        
        # Step 2: Super-Resolution
        if apply_super_resolution and self.model_manager.sr_model:
            start = time.time()
            result = self.model_manager.sr_model.enhance(result, outscale=sr_scale)
            stats['steps'].append('Super-Resolution')
            stats['timings']['super_resolution'] = time.time() - start
        
        # Step 3: Colorization
        if apply_colorization and self.model_manager.colorize_model:
            start = time.time()
            result = self.model_manager.colorize_model.colorize(result, render_factor=colorize_render_factor)
            stats['steps'].append('Colorization')
            stats['timings']['colorization'] = time.time() - start
        
        # Step 4: Detail Enhancement
        if apply_detail_enhancement and self.model_manager.detail_model:
            start = time.time()
            result = self.model_manager.detail_model.enhance_details(result, strength=detail_strength)
            stats['steps'].append('Detail Enhancement')
            stats['timings']['detail_enhancement'] = time.time() - start
        
        stats['output_shape'] = result.shape
        stats['total_time'] = sum(stats['timings'].values())
        
        return result, stats
    
    def restore_video(
        self,
        video_path: str,
        output_path: str,
        apply_super_resolution: bool = True,
        apply_denoising: bool = True,
        apply_colorization: bool = False,
        apply_detail_enhancement: bool = True,
        sr_scale: float = 2.0,
        denoise_strength: int = 10,
        colorize_render_factor: int = 25,
        detail_strength: float = 1.3,
        max_frames: Optional[int] = None,
        progress_callback=None
    ) -> Dict:
        """
        Process video frame by frame
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video
            apply_super_resolution: Enable super-resolution
            apply_denoising: Enable denoising
            apply_colorization: Enable colorization
            apply_detail_enhancement: Enable detail enhancement
            sr_scale: Super-resolution scale factor
            denoise_strength: Denoising strength
            colorize_render_factor: Colorization quality
            detail_strength: Detail enhancement strength
            max_frames: Maximum number of frames to process
            progress_callback: Function to call with progress updates
            
        Returns:
            Dictionary with processing statistics
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
        
        # Read first frame to determine output size
        ret, first_frame = cap.read()
        if not ret:
            raise ValueError("Cannot read first frame")
        
        # Process first frame to get output dimensions
        restored_first, _ = self.restore_image(
            first_frame,
            apply_super_resolution=apply_super_resolution,
            apply_denoising=apply_denoising,
            apply_colorization=apply_colorization,
            apply_detail_enhancement=apply_detail_enhancement,
            sr_scale=sr_scale,
            denoise_strength=denoise_strength,
            colorize_render_factor=colorize_render_factor,
            detail_strength=detail_strength
        )
        
        out_height, out_width = restored_first.shape[:2]
        
        # Initialize video writer - use MJPEG codec for better compatibility
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))
        
        # Write first frame
        out.write(restored_first)
        
        # Reset video capture
        cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
        
        stats = {
            'input_resolution': (width, height),
            'output_resolution': (out_width, out_height),
            'fps': fps,
            'total_frames': total_frames,
            'processed_frames': 1,
            'average_frame_time': 0
        }
        
        frame_times = []
        
        # Process remaining frames
        for frame_idx in range(1, total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            start = time.time()
            restored_frame, _ = self.restore_image(
                frame,
                apply_super_resolution=apply_super_resolution,
                apply_denoising=apply_denoising,
                apply_colorization=apply_colorization,
                apply_detail_enhancement=apply_detail_enhancement,
                sr_scale=sr_scale,
                denoise_strength=denoise_strength,
                colorize_render_factor=colorize_render_factor,
                detail_strength=detail_strength
            )
            frame_time = time.time() - start
            frame_times.append(frame_time)
            
            out.write(restored_frame)
            stats['processed_frames'] = frame_idx + 1
            
            if progress_callback:
                progress = (frame_idx + 1) / total_frames
                progress_callback(progress, frame_idx + 1, total_frames, frame_time)
        
        cap.release()
        out.release()
        
        stats['average_frame_time'] = np.mean(frame_times) if frame_times else 0
        stats['total_processing_time'] = sum(frame_times)
        
        return stats
    
    def batch_restore_images(
        self,
        input_dir: str,
        output_dir: str,
        **kwargs
    ) -> Dict:
        """
        Process all images in a directory
        
        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            **kwargs: Parameters to pass to restore_image
            
        Returns:
            Dictionary with batch processing statistics
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = [f for f in input_path.iterdir() if f.suffix.lower() in image_extensions]
        
        stats = {
            'total_images': len(image_files),
            'processed_images': 0,
            'failed_images': 0,
            'image_stats': []
        }
        
        for img_file in image_files:
            try:
                # Read image
                image = cv2.imread(str(img_file))
                if image is None:
                    stats['failed_images'] += 1
                    continue
                
                # Restore image
                restored, img_stats = self.restore_image(image, **kwargs)
                
                # Save result
                output_file = output_path / f"restored_{img_file.name}"
                cv2.imwrite(str(output_file), restored)
                
                img_stats['filename'] = img_file.name
                stats['image_stats'].append(img_stats)
                stats['processed_images'] += 1
                
            except Exception as e:
                print(f"Error processing {img_file.name}: {e}")
                stats['failed_images'] += 1
        
        return stats