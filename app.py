import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import tempfile
import os
from models import ModelManager
from restoration_pipeline import RestorationPipeline
from utils import (
    save_image, create_comparison_image, calculate_metrics,
    is_grayscale, resize_for_display, numpy_to_pil, format_time,
    get_video_info, create_output_directory
)

# Page config
st.set_page_config(
    page_title="AI Image & Video Restoration",
    page_icon="🎨",
    layout="wide"
)

# Initialize session state
if 'model_manager' not in st.session_state:
    st.session_state.model_manager = None
    st.session_state.pipeline = None
    st.session_state.models_loaded = False

def load_models():
    """Load all AI models"""
    with st.spinner("Loading AI models... This may take a few minutes on first run."):
        model_manager = ModelManager()
        models_loaded = model_manager.load_models(load_sr=True, load_colorize=True)
        pipeline = RestorationPipeline(model_manager)
        
        st.session_state.model_manager = model_manager
        st.session_state.pipeline = pipeline
        st.session_state.models_loaded = True
        
        return models_loaded

def process_image(image_file, params):
    """Process uploaded image"""
    # Read image
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Check if grayscale
    is_gray = is_grayscale(image)
    
    # Process image
    with st.spinner("Processing image..."):
        restored, stats = st.session_state.pipeline.restore_image(
            image,
            apply_super_resolution=params['super_resolution'],
            apply_denoising=params['denoising'],
            apply_colorization=params['colorization'] and is_gray,
            apply_detail_enhancement=params['detail_enhancement'],
            sr_scale=params['sr_scale'],
            denoise_strength=params['denoise_strength'],
            colorize_render_factor=params['colorize_render_factor'],
            detail_strength=params['detail_strength']
        )
    
    return image, restored, stats, is_gray

def process_video(video_file, params, max_frames=None):
    """Process uploaded video"""
    # Save uploaded video to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_input:
        tmp_input.write(video_file.read())
        input_path = tmp_input.name
    
    # Create output path
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_output:
        output_path = tmp_output.name
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    def progress_callback(progress, frame_num, total_frames, frame_time):
        progress_bar.progress(progress)
        status_text.text(f"Processing frame {frame_num}/{total_frames} ({progress*100:.1f}%) - {frame_time:.2f}s/frame")
    
    # Process video
    try:
        stats = st.session_state.pipeline.restore_video(
            input_path,
            output_path,
            apply_super_resolution=params['super_resolution'],
            apply_denoising=params['denoising'],
            apply_colorization=params['colorization'],
            apply_detail_enhancement=params['detail_enhancement'],
            sr_scale=params['sr_scale'],
            denoise_strength=params['denoise_strength'],
            colorize_render_factor=params['colorize_render_factor'],
            detail_strength=params['detail_strength'],
            max_frames=max_frames,
            progress_callback=progress_callback
        )
        
        progress_bar.empty()
        status_text.empty()
        
        return output_path, stats
    
    finally:
        # Cleanup input file
        try:
            os.unlink(input_path)
        except:
            pass

# Main UI
st.title("🎨 AI Image & Video Restoration")
st.markdown("### Restore and enhance old or low-quality images and videos using state-of-the-art AI")

# Sidebar for model loading and settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    if not st.session_state.models_loaded:
        if st.button("🚀 Load AI Models", type="primary"):
            models_status = load_models()
            if st.session_state.models_loaded:
                st.success("✅ Models loaded successfully!")
                st.json(models_status)
    else:
        st.success("✅ Models ready")
        
        st.markdown("---")
        st.header("🎛️ Processing Parameters")
        
        # Processing options
        st.subheader("Enhancement Options")
        super_resolution = st.checkbox("Super-Resolution", value=True, 
                                      help="Upscale image resolution using AI")
        denoising = st.checkbox("Denoising", value=True,
                               help="Remove noise and compression artifacts")
        colorization = st.checkbox("Colorization", value=False,
                                  help="Colorize black & white images")
        detail_enhancement = st.checkbox("Detail Enhancement", value=True,
                                        help="Enhance fine details and sharpness")
        
        st.markdown("---")
        st.subheader("Advanced Parameters")
        
        if super_resolution:
            sr_scale = st.slider("SR Scale Factor", 2.0, 4.0, 4.0, 0.5,
                               help="How much to upscale (higher = more processing time)")
        else:
            sr_scale = 2.0
        
        if denoising:
            denoise_strength = st.slider("Denoise Strength", 1, 30, 10,
                                        help="Higher = more aggressive denoising")
        else:
            denoise_strength = 10
        
        if colorization:
            colorize_render_factor = st.slider("Colorization Quality", 10, 40, 35,
                                              help="Higher = better quality but slower")
        else:
            colorize_render_factor = 35
        
        if detail_enhancement:
            detail_strength = st.slider("Detail Strength", 1.0, 2.0, 1.5, 0.1,
                                       help="How much to enhance details")
        else:
            detail_strength = 1.5
        
        params = {
            'super_resolution': super_resolution,
            'denoising': denoising,
            'colorization': colorization,
            'detail_enhancement': detail_enhancement,
            'sr_scale': sr_scale,
            'denoise_strength': denoise_strength,
            'colorize_render_factor': colorize_render_factor,
            'detail_strength': detail_strength
        }

# Main content
if not st.session_state.models_loaded:
    st.info("👈 Click 'Load AI Models' in the sidebar to get started")
else:
    # Tab selection
    tab1, tab2 = st.tabs(["📷 Image Restoration", "🎬 Video Restoration"])
    
    with tab1:
        st.header("Image Restoration")
        
        uploaded_image = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp'])
        
        if uploaded_image is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                original_image, restored_image, stats, is_gray = process_image(uploaded_image, params)
                
                # Display original
                display_original = resize_for_display(original_image, max_width=800, max_height=600)
                st.image(cv2.cvtColor(display_original, cv2.COLOR_BGR2RGB), use_container_width=True)
                
                st.metric("Original Resolution", f"{original_image.shape[1]}x{original_image.shape[0]}")
                if is_gray:
                    st.info("ℹ️ Grayscale image detected")
            
            with col2:
                st.subheader("Restored Image")
                display_restored = resize_for_display(restored_image, max_width=800, max_height=600)
                st.image(cv2.cvtColor(display_restored, cv2.COLOR_BGR2RGB), use_container_width=True)
                
                st.metric("Restored Resolution", f"{restored_image.shape[1]}x{restored_image.shape[0]}")
                st.metric("Processing Time", format_time(stats['total_time']))
            
            # Processing statistics
            with st.expander("📊 Processing Statistics"):
                st.json(stats)
            
            # Quality metrics
            with st.expander("📈 Quality Metrics"):
                metrics = calculate_metrics(original_image, restored_image)
                col_m1, col_m2, col_m3 = st.columns(3)
                
                with col_m1:
                    if metrics['psnr']:
                        st.metric("PSNR", f"{metrics['psnr']:.2f} dB")
                
                with col_m2:
                    st.metric("MSE", f"{metrics['mse']:.2f}")
                
                with col_m3:
                    if metrics['sharpness_improvement']:
                        improvement = (metrics['sharpness_improvement'] - 1) * 100
                        st.metric("Sharpness Gain", f"+{improvement:.1f}%")
            
            # Download buttons
            st.markdown("---")
            col_d1, col_d2 = st.columns(2)
            
            with col_d1:
                # Save restored image
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                    tmp_path = tmp.name
                    cv2.imwrite(tmp_path, restored_image)
                
                with open(tmp_path, 'rb') as f:
                    st.download_button(
                        label="⬇️ Download Restored Image",
                        data=f,
                        file_name=f"restored_{uploaded_image.name}",
                        mime="image/png"
                    )
                try:
                    os.unlink(tmp_path)
                except PermissionError:
                    pass  # File will be cleaned up by temp directory
            
            with col_d2:
                # Save comparison image
                comparison = create_comparison_image(original_image, restored_image)
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                    tmp_path = tmp.name
                    cv2.imwrite(tmp_path, comparison)
                
                with open(tmp_path, 'rb') as f:
                    st.download_button(
                        label="⬇️ Download Comparison",
                        data=f,
                        file_name=f"comparison_{uploaded_image.name}",
                        mime="image/png"
                    )
                try:
                    os.unlink(tmp_path)
                except PermissionError:
                    pass  # File will be cleaned up by temp directory
    
    with tab2:
        st.header("Video Restoration")
        st.warning("⚠️ Video processing is computationally intensive and may take several minutes.")
        
        uploaded_video = st.file_uploader("Upload a video", type=['mp4', 'avi', 'mov', 'mkv'])
        
        if uploaded_video is not None:
            # Save video temporarily to get info
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(uploaded_video.read())
                video_path = tmp.name
            
            video_info = get_video_info(video_path)
            os.unlink(video_path)
            
            # Reset file pointer
            uploaded_video.seek(0)
            
            # Display video info
            col_v1, col_v2, col_v3, col_v4 = st.columns(4)
            with col_v1:
                st.metric("Resolution", f"{video_info['width']}x{video_info['height']}")
            with col_v2:
                st.metric("FPS", video_info['fps'])
            with col_v3:
                st.metric("Total Frames", video_info['frame_count'])
            with col_v4:
                if video_info['duration_seconds']:
                    st.metric("Duration", format_time(video_info['duration_seconds']))
            
            # Frame limit option
            process_all_frames = st.checkbox("Process all frames", value=False)
            if not process_all_frames:
                max_frames = st.number_input("Maximum frames to process", 
                                            min_value=1, 
                                            max_value=video_info['frame_count'],
                                            value=min(300, video_info['frame_count']),
                                            help="Limit frames for faster testing")
            else:
                max_frames = None
            
            if st.button("🎬 Start Video Restoration", type="primary"):
                output_path, stats = process_video(uploaded_video, params, max_frames)
                
                st.success("✅ Video processing complete!")
                
                # Display statistics
                col_s1, col_s2, col_s3 = st.columns(3)
                with col_s1:
                    st.metric("Processed Frames", stats['processed_frames'])
                with col_s2:
                    st.metric("Avg Frame Time", f"{stats['average_frame_time']:.2f}s")
                with col_s3:
                    st.metric("Total Time", format_time(stats['total_processing_time']))
                
                # Show output video
                st.video(output_path)
                
                # Download button
                with open(output_path, 'rb') as f:
                    st.download_button(
                        label="⬇️ Download Restored Video",
                        data=f,
                        file_name=f"restored_{uploaded_video.name}",
                        mime="video/mp4"
                    )
                
                # Cleanup
                try:
                    os.unlink(output_path)
                except:
                    pass

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with state-of-the-art AI models: Real-ESRGAN, DDColor, and OpenCV</p>
    <p>🔬 Computer Vision | 🤖 Deep Learning | 🎨 Image Processing</p>
</div>
""", unsafe_allow_html=True)