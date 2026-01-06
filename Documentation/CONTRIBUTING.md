# Contributing to AI Image & Video Restoration

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to this project.

## 🎯 Ways to Contribute

### 1. Report Bugs
- Use GitHub Issues to report bugs
- Include detailed reproduction steps
- Provide system information (OS, GPU, Python version)
- Include error messages and logs

### 2. Suggest Features
- Open an issue with the "enhancement" label
- Describe the use case and expected behavior
- Explain why this feature would be valuable

### 3. Submit Code
- Fork the repository
- Create a feature branch
- Make your changes
- Submit a pull request

### 4. Improve Documentation
- Fix typos or unclear explanations
- Add examples or use cases
- Improve code comments

### 5. Share Results
- Share before/after examples
- Provide performance benchmarks
- Document use cases

## 🚀 Development Setup

### 1. Fork and Clone

```bash
git clone https://github.com/AliTraks/ai-restoration.git
cd ai-restoration
```

### 2. Create Development Environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies
```

### 3. Install Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install
```

## 📝 Coding Standards

### Python Style Guide
- Follow PEP 8
- Use type hints where appropriate
- Write docstrings for all functions/classes
- Keep functions focused and modular

### Example:

```python
def restore_image(
    image: np.ndarray,
    apply_super_resolution: bool = True,
    sr_scale: float = 4.0
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Restore and enhance an image using AI models.
    
    Args:
        image: Input image as numpy array (BGR format)
        apply_super_resolution: Whether to apply super-resolution
        sr_scale: Upscaling factor (2.0-4.0)
        
    Returns:
        Tuple of (restored_image, processing_statistics)
        
    Raises:
        ValueError: If image is None or invalid scale factor
    """
    # Implementation
    pass
```

### Code Organization
- Models: `models.py`
- Pipeline: `restoration_pipeline.py`
- Utilities: `utils.py`
- Configuration: `config.py`
- UI: `app.py`
- CLI: `restore.py`

## 🧪 Testing

### Run Tests

```bash
pytest tests/
```

### Write Tests

```python
# tests/test_pipeline.py
import pytest
from restoration_pipeline import RestorationPipeline

def test_restore_image():
    """Test basic image restoration"""
    # Create test image
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Initialize pipeline
    pipeline = RestorationPipeline(model_manager)
    
    # Restore image
    restored, stats = pipeline.restore_image(test_image)
    
    # Assertions
    assert restored is not None
    assert restored.shape[0] >= test_image.shape[0]
```

## 📋 Pull Request Process

### 1. Before Submitting

- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] Documentation is updated
- [ ] Commit messages are clear
- [ ] Branch is up to date with main

### 2. Commit Message Format

```
type(scope): brief description

Detailed explanation of changes

Fixes #issue_number
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test additions/changes
- `perf`: Performance improvements

Example:
```
feat(colorization): add batch colorization support

Implement batch processing for colorizing multiple images
simultaneously, improving throughput by 3x.

Fixes #42
```

### 3. Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Testing
How have these changes been tested?

## Screenshots (if applicable)
Before/after comparisons

## Checklist
- [ ] Code follows style guidelines
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] No breaking changes
```

## 🎨 Areas for Contribution

### High Priority
1. **Additional AI Models**
   - Face restoration (GFPGAN, CodeFormer)
   - Scratch and artifact removal
   - Old film restoration
   - Alternative colorization models

2. **Performance Optimization**
   - Multi-GPU support
   - Batch processing optimization
   - Memory efficiency improvements
   - Quantization for faster inference

3. **Quality Metrics**
   - LPIPS (perceptual similarity)
   - FID (Fréchet Inception Distance)
   - Automated quality assessment

4. **Video Processing**
   - Temporal consistency
   - Frame interpolation
   - Audio preservation

### Medium Priority
1. **UI Improvements**
   - Parameter presets
   - History tracking
   - Real-time preview

2. **Format Support**
   - RAW image formats
   - Additional video codecs
   - HDR support

3. **Cloud Integration**
   - S3/Google Cloud Storage
   - Batch API endpoint
   - Web service deployment

### Documentation
1. **Tutorials**
   - Video walkthroughs
   - Use case examples
   - Parameter guides

2. **API Documentation**
   - Detailed function references
   - Integration examples
   - Best practices

## 🐛 Bug Reports

### Required Information
1. **System Information**
   - OS and version
   - Python version
   - GPU model and driver version
   - CUDA version

2. **Bug Description**
   - What were you trying to do?
   - What happened instead?
   - Can you reproduce it?

3. **Code to Reproduce**
```python
# Minimal example that reproduces the issue
```

4. **Error Messages**
```
Full error traceback
```

## 💬 Communication

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and general discussion
- **Pull Requests**: Code contributions
- **Email**: [ali1313artin@gmail.com] for sensitive matters

## 🏆 Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Credited in release notes
- Mentioned in relevant documentation

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## ❓ Questions?

Feel free to:
- Open an issue with the "question" label
- Start a GitHub Discussion
- Contact the maintainers directly

Thank you for contributing to AI Image & Video Restoration! 🎉