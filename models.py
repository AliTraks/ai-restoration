import torch
import cv2
import numpy as np
from PIL import Image
from typing import Optional
import requests
from io import BytesIO

class SuperResolutionModel:
    def __init__(self, model_name: str = "realesrgan-x4plus"):
        self.model_name = model_name
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load(self):
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            
            if self.model_name == "realesrgan-x4plus":
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
                netscale = 4
                model_path = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
            elif self.model_name == "realesrgan-x2plus":
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
                netscale = 2
                model_path = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
            else:
                raise ValueError(f"Unknown model: {self.model_name}")
            
            self.model = RealESRGANer(
                scale=netscale,
                model_path=model_path,
                model=model,
                tile=400,
                tile_pad=10,
                pre_pad=0,
                half=True if self.device.type == "cuda" else False,
                device=self.device
            )
            return True
        except Exception as e:
            print(f"Error loading super-resolution model: {e}")
            return False
    
    def enhance(self, image: np.ndarray, outscale: float = 4.0) -> np.ndarray:
        if self.model is None:
            return image
        try:
            output, _ = self.model.enhance(image, outscale=outscale)
            return output
        except Exception as e:
            print(f"Super-resolution error: {e}")
            return image


class DenoisingModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def denoise(self, image: np.ndarray, strength: int = 10) -> np.ndarray:
        try:
            if len(image.shape) == 3:
                denoised = cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)
            else:
                denoised = cv2.fastNlMeansDenoising(image, None, strength, 7, 21)
            return denoised
        except Exception as e:
            print(f"Denoising error: {e}")
            return image


class ColorizationModel:
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load(self):
        try:
            from transformers import AutoModelForImageSegmentation, AutoImageProcessor
            import torchvision.transforms as transforms
            
            # Load DDColor - state-of-the-art colorization model
            model_name = "piddnad/ddcolor-paper"
            
            print(f"Loading colorization model from Hugging Face: {model_name}")
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModelForImageSegmentation.from_pretrained(model_name)
            self.model = self.model.to(self.device)
            self.model.eval()
            
            return True
        except Exception as e:
            print(f"Error loading colorization model: {e}")
            print("Falling back to basic grayscale-to-RGB conversion")
            return False
    
    def colorize(self, image: np.ndarray, render_factor: int = 35) -> np.ndarray:
        """
        Colorize grayscale image using modern deep learning
        
        Args:
            image: Input image (grayscale or RGB)
            render_factor: Quality parameter (10-40, higher = better quality)
        """
        if self.model is None:
            # Fallback: simple conversion to RGB if model not loaded
            if len(image.shape) == 2:
                return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            return image
        
        try:
            # Convert to grayscale if needed for colorization
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Convert to PIL RGB (grayscale expanded to 3 channels)
            gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            pil_image = Image.fromarray(gray_rgb)
            
            # Resize based on render_factor for quality/speed trade-off
            # render_factor: 10-40 maps to size multiplier
            size_multiplier = render_factor / 35.0
            target_size = int(512 * size_multiplier)
            
            original_size = pil_image.size
            pil_image_resized = pil_image.resize((target_size, target_size), Image.LANCZOS)
            
            # Process with model
            with torch.no_grad():
                inputs = self.processor(images=pil_image_resized, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = self.model(**inputs)
                
                # Get the colorized output
                if hasattr(outputs, 'logits'):
                    result_tensor = outputs.logits
                else:
                    result_tensor = outputs[0]
                
                # Post-process
                result_tensor = torch.nn.functional.interpolate(
                    result_tensor,
                    size=(target_size, target_size),
                    mode='bilinear',
                    align_corners=False
                )
                
                # Convert to numpy
                result_np = result_tensor.squeeze().cpu().numpy()
                
                # Normalize to 0-255 range
                if result_np.max() > 1.0:
                    result_np = (result_np * 255).astype(np.uint8)
                else:
                    result_np = (result_np * 255).astype(np.uint8)
                
                # Ensure correct shape (H, W, C)
                if result_np.ndim == 3 and result_np.shape[0] == 3:
                    result_np = np.transpose(result_np, (1, 2, 0))
                
                # Convert to PIL and resize back to original size
                result_pil = Image.fromarray(result_np)
                result_pil = result_pil.resize(original_size, Image.LANCZOS)
                
                # Convert back to numpy BGR
                result_np = np.array(result_pil)
                if result_np.shape[2] == 3:
                    result_np = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)
                
                return result_np
                
        except Exception as e:
            print(f"Colorization error: {e}")
            print("Falling back to grayscale-to-RGB conversion")
            # Fallback: simple conversion
            if len(image.shape) == 2:
                return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            return image


class DetailEnhancementModel:
    def __init__(self):
        pass
    
    def enhance_details(self, image: np.ndarray, strength: float = 1.5) -> np.ndarray:
        try:
            # Apply unsharp masking for detail enhancement
            gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
            unsharp_image = cv2.addWeighted(image, strength, gaussian, -(strength - 1), 0)
            return np.clip(unsharp_image, 0, 255).astype(np.uint8)
        except Exception as e:
            print(f"Detail enhancement error: {e}")
            return image


class ModelManager:
    def __init__(self):
        self.sr_model = None
        self.denoise_model = None
        self.colorize_model = None
        self.detail_model = None
        
    def load_models(self, load_sr: bool = True, load_colorize: bool = True):
        models_loaded = {}
        
        if load_sr:
            print("Loading super-resolution model...")
            self.sr_model = SuperResolutionModel()
            models_loaded['super_resolution'] = self.sr_model.load()
        
        print("Loading denoising model...")
        self.denoise_model = DenoisingModel()
        models_loaded['denoising'] = True
        
        if load_colorize:
            print("Loading colorization model...")
            self.colorize_model = ColorizationModel()
            models_loaded['colorization'] = self.colorize_model.load()
        
        print("Loading detail enhancement model...")
        self.detail_model = DetailEnhancementModel()
        models_loaded['detail_enhancement'] = True
        
        return models_loaded