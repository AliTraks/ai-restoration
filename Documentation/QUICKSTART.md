# 🚀 Quick Start Guide

Get up and running with AI Image & Video Restoration in 5 minutes.

## ⚡ Quick Installation

### Option 1: Standard Installation (Recommended)

```bash
# Clone repository
git clone https://github.com/AliTraks/ai-restoration.git
cd ai-restoration

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the web interface
streamlit run app.py
```

### Option 2: pip Installation

```bash
pip install ai-image-video-restoration
restore --help
```

## 🎯 Basic Usage

### 1. Web Interface (Easiest)

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser and:
1. Click "Load AI Models" in the sidebar
2. Upload an image or video
3. Adjust parameters (optional)
4. Download your restored media

### 2. Command Line

```bash
# Restore a single image
python restore.py --input photo.jpg --output restored.jpg

# Colorize a black and white photo
python restore.py --input bw_photo.jpg --output colorized.jpg --colorize

# Process a video (first 300 frames)
python restore.py --input video.mp4 --output restored.mp4 --video --max-frames 300

# Batch process a directory
python restore.py --input ./photos --output ./restored --batch
```

### 3. Python API

```python
from models import ModelManager
from restoration_pipeline import RestorationPipeline
import cv2

# Load models
model_manager = ModelManager()
model_manager.load_models()
pipeline = RestorationPipeline(model_manager)

# Restore image
image = cv2.imread('photo.jpg')
restored, stats = pipeline.restore_image(image)
cv2.imwrite('restored.jpg', restored)
```

## 📊 Example Results

### Before/After Comparison

**Original (512x512)** → **Restored (2048x2048)**

Processing steps:
- ✓ Super-resolution (4x upscale)
- ✓ Denoising
- ✓ Detail enhancement

**Time**: ~3.5 seconds on RTX 3080

## 🎛️ Common Use Cases

### High-Quality Photo Restoration
```bash
python restore.py --input old_photo.jpg --output restored.jpg --sr-scale 4.0 --denoise-strength 15
```

### Quick Video Preview
```bash
python restore.py --input video.mp4 --output preview.mp4 --video --sr-scale 2.0 --max-frames 100
```

### Colorize Historical Photos
```bash
python restore.py --input bw_archive/ --output colorized/ --batch --colorize --colorize-quality 35
```

### Minimal Processing (Fast)
```bash
python restore.py --input photo.jpg --output quick.jpg --no-sr --denoise-strength 5
```

## ⚙️ Parameter Recommendations

### For Historical Photos
- Super-resolution: 4x
- Denoising: 15-20
- Colorization: On (if B&W)
- Detail strength: 1.5

### For Videos
- Super-resolution: 2x (faster)
- Denoising: 10
- Detail strength: 1.3
- Process in batches: 300-500 frames

### For Modern Low-Res Images
- Super-resolution: 2x
- Denoising: 5-10
- Detail strength: 1.2

## 🐛 Troubleshooting

### "Out of memory" Error
```python
# Solution 1: Reduce scale factor
--sr-scale 2.0  # instead of 4.0

# Solution 2: Disable super-resolution for large images
--no-sr
```

### Slow Processing
```python
# Check GPU availability
import torch
print(torch.cuda.is_available())  # Should be True

# Use lower quality settings
--sr-scale 2.0 --denoise-strength 5
```

### Models Not Loading
```bash
# Clear cache and reinstall
rm -rf ~/.cache/torch/hub
pip install --upgrade basicsr realesrgan
```

## 📚 Next Steps

1. **Explore Examples**: Run `python example_usage.py`
2. **Read Full Documentation**: Check `README.md`
3. **Customize Parameters**: Edit `config.py`
4. **Advanced Usage**: See API documentation

## 🔗 Resources

- **Documentation**: Full README.md
- **Examples**: `example_usage.py`
- **Configuration**: `config.py`
- **CLI Help**: `python restore.py --help`

## 💡 Pro Tips

1. **First run is slow**: Models download (~300MB total) on first use
2. **GPU is essential**: CPU processing is 5-10x slower
3. **Start with small images**: Test parameters before processing large batches
4. **Save comparisons**: Use `--comparison` flag to see before/after
5. **Monitor memory**: Use `nvidia-smi` to watch GPU memory usage

## 🎉 You're Ready!

Your AI restoration system is set up. Start with the web interface for the best experience, or use the CLI for batch processing.

**First command to try**:
```bash
streamlit run app.py
```

Happy restoring! 🎨