"""
Configuration file for AI Image & Video Restoration System
Centralized configuration for easy customization and deployment
"""

import torch
from pathlib import Path

# ============================================================================
# SYSTEM CONFIGURATION
# ============================================================================

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_HALF_PRECISION = True if DEVICE == "cuda" else False

# Directory paths
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
MODELS_DIR = BASE_DIR / "models" / "weights"
TEMP_DIR = BASE_DIR / "temp"

# Create directories if they don't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Super-Resolution Models
SR_MODELS = {
    "realesrgan-x4plus": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "scale": 4,
        "description": "Best quality, 4x upscaling"
    },
    "realesrgan-x2plus": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        "scale": 2,
        "description": "Faster processing, 2x upscaling"
    }
}

# Default super-resolution model
DEFAULT_SR_MODEL = "realesrgan-x4plus"

# Colorization settings
COLORIZATION_MODELS = {
    "ddcolor": {
        "model_name": "piddnad/ddcolor-paper",
        "description": "State-of-the-art transformer-based colorization",
        "default_render_factor": 35
    },
    "ddcolor-artistic": {
        "model_name": "piddnad/ddcolor-artistic", 
        "description": "Artistic colorization with vibrant colors",
        "default_render_factor": 35
    }
}

DEFAULT_COLORIZATION_MODEL = "ddcolor"

# ============================================================================
# PROCESSING DEFAULTS
# ============================================================================

# Default enhancement parameters
DEFAULT_PARAMS = {
    # Enable/disable operations
    "apply_super_resolution": True,
    "apply_denoising": True,
    "apply_colorization": False,  # Auto-detect grayscale
    "apply_detail_enhancement": True,
    
    # Parameter values
    "sr_scale": 4.0,
    "denoise_strength": 10,
    "colorize_render_factor": 35,
    "detail_strength": 1.5
}

# Parameter ranges and constraints
PARAM_RANGES = {
    "sr_scale": {
        "min": 2.0,
        "max": 4.0,
        "step": 0.5,
        "description": "Super-resolution upscaling factor"
    },
    "denoise_strength": {
        "min": 1,
        "max": 30,
        "step": 1,
        "description": "Denoising intensity (higher = more aggressive)"
    },
    "colorize_render_factor": {
        "min": 10,
        "max": 40,
        "step": 5,
        "description": "Colorization quality (higher = better but slower)"
    },
    "detail_strength": {
        "min": 1.0,
        "max": 2.0,
        "step": 0.1,
        "description": "Detail enhancement strength"
    }
}

# ============================================================================
# VIDEO PROCESSING
# ============================================================================

VIDEO_CONFIG = {
    # Default video settings
    "default_fps": 30,
    "default_codec": "mp4v",
    "max_frames_preview": 300,  # For quick preview processing
    
    # Recommended settings for different video types
    "presets": {
        "fast": {
            "sr_scale": 2.0,
            "denoise_strength": 5,
            "detail_strength": 1.2,
            "description": "Fast processing, moderate quality"
        },
        "balanced": {
            "sr_scale": 2.0,
            "denoise_strength": 10,
            "detail_strength": 1.5,
            "description": "Balanced quality and speed"
        },
        "quality": {
            "sr_scale": 4.0,
            "denoise_strength": 15,
            "detail_strength": 1.8,
            "description": "Best quality, slower processing"
        }
    }
}

# ============================================================================
# BATCH PROCESSING
# ============================================================================

BATCH_CONFIG = {
    # Supported image formats
    "image_extensions": [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"],
    
    # Supported video formats
    "video_extensions": [".mp4", ".avi", ".mov", ".mkv"],
    
    # Batch processing settings
    "max_workers": 1,  # Sequential processing (models are GPU-bound)
    "skip_existing": True,  # Skip already processed files
    "create_comparisons": True,  # Generate before/after comparisons
}

# ============================================================================
# QUALITY METRICS
# ============================================================================

METRICS_CONFIG = {
    "calculate_psnr": True,
    "calculate_ssim": True,
    "calculate_sharpness": True,
    "save_metrics": True,  # Save metrics to JSON file
}

# ============================================================================
# UI CONFIGURATION
# ============================================================================

UI_CONFIG = {
    # Streamlit settings
    "page_title": "AI Image & Video Restoration",
    "page_icon": "🎨",
    "layout": "wide",
    
    # Display settings
    "max_display_width": 1920,
    "max_display_height": 1080,
    "comparison_border_width": 5,
    
    # Performance settings
    "enable_caching": True,
    "cache_ttl": 3600,  # Cache timeout in seconds
}

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOGGING_CONFIG = {
    "log_level": "INFO",  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    "log_file": BASE_DIR / "restoration.log",
    "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_to_console": True,
    "log_to_file": True,
}

# ============================================================================
# OPTIMIZATION SETTINGS
# ============================================================================

OPTIMIZATION_CONFIG = {
    # Memory management
    "tile_size": 400,  # Tile size for processing large images
    "tile_padding": 10,
    
    # Processing optimizations
    "use_fp16": USE_HALF_PRECISION,
    "benchmark": True,  # Enable CUDNN benchmarking for speed
    "deterministic": False,  # Disable for better performance
    
    # Batch size for video frames (if processing multiple frames at once)
    "video_batch_size": 1,  # Keep at 1 for memory efficiency
}

# ============================================================================
# ADVANCED SETTINGS
# ============================================================================

ADVANCED_CONFIG = {
    # Model-specific settings
    "sr_tile_size": 400,
    "sr_tile_pad": 10,
    "sr_pre_pad": 0,
    
    # Denoising settings
    "denoise_window_size": 7,
    "denoise_template_size": 21,
    
    # Detail enhancement
    "detail_gaussian_sigma": 2.0,
    
    # Colorization
    "colorize_min_size": 256,  # Minimum dimension for colorization
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_preset_params(preset_name: str) -> dict:
    """Get parameter configuration for a preset"""
    if preset_name in VIDEO_CONFIG["presets"]:
        return VIDEO_CONFIG["presets"][preset_name]
    return DEFAULT_PARAMS

def validate_params(params: dict) -> dict:
    """Validate and clip parameters to allowed ranges"""
    validated = params.copy()
    
    for param, value in params.items():
        if param in PARAM_RANGES:
            range_info = PARAM_RANGES[param]
            validated[param] = max(range_info["min"], 
                                  min(range_info["max"], value))
    
    return validated

def get_model_info(model_type: str = "sr") -> dict:
    """Get information about available models"""
    if model_type == "sr":
        return SR_MODELS
    elif model_type == "colorization":
        return COLORIZATION_MODELS
    return {}

# ============================================================================
# CONFIGURATION EXPORT
# ============================================================================

# Export main configuration object
CONFIG = {
    "system": {
        "device": DEVICE,
        "use_half_precision": USE_HALF_PRECISION,
        "output_dir": OUTPUT_DIR,
        "models_dir": MODELS_DIR,
        "temp_dir": TEMP_DIR,
    },
    "models": {
        "sr_models": SR_MODELS,
        "default_sr_model": DEFAULT_SR_MODEL,
        "colorization_models": COLORIZATION_MODELS,
        "default_colorization_model": DEFAULT_COLORIZATION_MODEL,
    },
    "defaults": DEFAULT_PARAMS,
    "ranges": PARAM_RANGES,
    "video": VIDEO_CONFIG,
    "batch": BATCH_CONFIG,
    "metrics": METRICS_CONFIG,
    "ui": UI_CONFIG,
    "logging": LOGGING_CONFIG,
    "optimization": OPTIMIZATION_CONFIG,
    "advanced": ADVANCED_CONFIG,
}

if __name__ == "__main__":
    import json
    print("Current Configuration:")
    print(json.dumps(CONFIG, indent=2, default=str))