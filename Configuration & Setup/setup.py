from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ai-image-video-restoration",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Production-grade AI system for image and video restoration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ai-restoration",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Multimedia :: Video",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "opencv-python>=4.8.0",
        "pillow>=10.0.0",
        "streamlit>=1.28.0",
        "matplotlib>=3.7.0",
        "basicsr>=1.4.2",
        "realesrgan>=0.3.0",
        "facexlib>=0.3.0",
        "gfpgan>=1.3.8",
        "transformers>=4.35.0",
        "accelerate>=0.24.0",
        "scikit-image>=0.21.0",
        "tqdm>=4.66.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "restore=restore:main",
        ],
    },
    include_package_data=True,
    keywords="image-restoration video-restoration super-resolution colorization denoising computer-vision deep-learning ai",
)