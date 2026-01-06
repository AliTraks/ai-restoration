import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional
import matplotlib.pyplot as plt

def load_image(image_path: str) -> np.ndarray:
    """Load image from file path"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot load image from {image_path}")
    return image

def save_image(image: np.ndarray, output_path: str):
    """Save image to file path"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, image)

def create_comparison_image(
    original: np.ndarray,
    restored: np.ndarray,
    title_original: str = "Original",
    title_restored: str = "Restored"
) -> np.ndarray:
    """Create side-by-side comparison of original and restored images"""
    
    # Resize images to same height if needed
    h_orig, w_orig = original.shape[:2]
    h_rest, w_rest = restored.shape[:2]
    
    target_height = max(h_orig, h_rest)
    
    if h_orig != target_height:
        scale = target_height / h_orig
        original = cv2.resize(original, (int(w_orig * scale), target_height))
    
    if h_rest != target_height:
        scale = target_height / h_rest
        restored = cv2.resize(restored, (int(w_rest * scale), target_height))
    
    # Add text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    thickness = 3
    color = (255, 255, 255)
    
    original_labeled = original.copy()
    restored_labeled = restored.copy()
    
    # Get text size for positioning
    text_size_orig = cv2.getTextSize(title_original, font, font_scale, thickness)[0]
    text_size_rest = cv2.getTextSize(title_restored, font, font_scale, thickness)[0]
    
    # Position text at top center
    text_x_orig = (original.shape[1] - text_size_orig[0]) // 2
    text_x_rest = (restored.shape[1] - text_size_rest[0]) // 2
    text_y = 50
    
    cv2.putText(original_labeled, title_original, (text_x_orig, text_y), 
                font, font_scale, color, thickness, cv2.LINE_AA)
    cv2.putText(restored_labeled, title_restored, (text_x_rest, text_y), 
                font, font_scale, color, thickness, cv2.LINE_AA)
    
    # Add black border between images
    border_width = 5
    border = np.zeros((target_height, border_width, 3), dtype=np.uint8)
    
    # Concatenate horizontally
    comparison = np.hstack([original_labeled, border, restored_labeled])
    
    return comparison

def save_comparison(
    original: np.ndarray,
    restored: np.ndarray,
    output_path: str,
    title_original: str = "Original",
    title_restored: str = "Restored"
):
    """Create and save comparison image"""
    comparison = create_comparison_image(original, restored, title_original, title_restored)
    save_image(comparison, output_path)

def calculate_metrics(original: np.ndarray, restored: np.ndarray) -> dict:
    """Calculate quality metrics between original and restored images"""
    
    # Resize to same dimensions for comparison
    if original.shape != restored.shape:
        restored_resized = cv2.resize(restored, (original.shape[1], original.shape[0]))
    else:
        restored_resized = restored
    
    # Convert to grayscale for some metrics
    if len(original.shape) == 3:
        orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        rest_gray = cv2.cvtColor(restored_resized, cv2.COLOR_BGR2GRAY)
    else:
        orig_gray = original
        rest_gray = restored_resized
    
    # Calculate PSNR
    try:
        psnr = cv2.PSNR(original, restored_resized)
    except:
        psnr = None
    
    # Calculate MSE
    mse = np.mean((original.astype(float) - restored_resized.astype(float)) ** 2)
    
    # Calculate sharpness (using Laplacian variance)
    orig_sharpness = cv2.Laplacian(orig_gray, cv2.CV_64F).var()
    rest_sharpness = cv2.Laplacian(rest_gray, cv2.CV_64F).var()
    
    metrics = {
        'psnr': psnr,
        'mse': mse,
        'original_sharpness': orig_sharpness,
        'restored_sharpness': rest_sharpness,
        'sharpness_improvement': rest_sharpness / orig_sharpness if orig_sharpness > 0 else None
    }
    
    return metrics

def extract_video_frame(video_path: str, frame_number: int = 0) -> np.ndarray:
    """Extract a specific frame from video"""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError(f"Cannot extract frame {frame_number} from {video_path}")
    
    return frame

def get_video_info(video_path: str) -> dict:
    """Get video metadata"""
    cap = cv2.VideoCapture(video_path)
    
    info = {
        'fps': int(cap.get(cv2.CAP_PROP_FPS)),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'duration_seconds': None
    }
    
    if info['fps'] > 0:
        info['duration_seconds'] = info['frame_count'] / info['fps']
    
    cap.release()
    return info

def is_grayscale(image: np.ndarray) -> bool:
    """Check if image is grayscale"""
    if len(image.shape) == 2:
        return True
    if len(image.shape) == 3 and image.shape[2] == 1:
        return True
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Check if all channels are identical
        return np.allclose(image[:, :, 0], image[:, :, 1]) and np.allclose(image[:, :, 1], image[:, :, 2])
    return False

def resize_for_display(image: np.ndarray, max_width: int = 1920, max_height: int = 1080) -> np.ndarray:
    """Resize image for display while maintaining aspect ratio"""
    h, w = image.shape[:2]
    
    if w <= max_width and h <= max_height:
        return image
    
    scale = min(max_width / w, max_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

def numpy_to_pil(image: np.ndarray) -> Image.Image:
    """Convert numpy array to PIL Image"""
    if len(image.shape) == 2:
        return Image.fromarray(image)
    else:
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

def pil_to_numpy(image: Image.Image) -> np.ndarray:
    """Convert PIL Image to numpy array"""
    arr = np.array(image)
    if len(arr.shape) == 3 and arr.shape[2] == 3:
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return arr

def format_time(seconds: float) -> str:
    """Format seconds into human-readable time string"""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.1f}s"

def create_output_directory(base_path: str = "output") -> Path:
    """Create timestamped output directory"""
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_path) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir