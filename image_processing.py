from PIL import Image
import tensorflow as tf
import numpy as np
from io import BytesIO
import logging

logger = logging.getLogger(__name__)


def smart_crop_to_square(image: Image.Image) -> Image.Image:
    """Crop image to square focusing on center"""
    width, height = image.size
    
    if width == height:
        return image
    
    # Crop to square using center
    size = min(width, height)
    left = (width - size) // 2
    top = (height - size) // 2
    right = left + size
    bottom = top + size
    
    return image.crop((left, top, right, bottom))


def preprocess_image(image: Image.Image, target_size: int = 448) -> Image.Image:
    """
    Preprocess image for model input
    
    Args:
        image: PIL Image
        target_size: Target size for square image (default 448)
    
    Returns:
        Preprocessed PIL Image
    """
    try:
        # Ensure RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Smart crop to square
        image = smart_crop_to_square(image)
        
        # Resize to target size
        image = image.resize((target_size, target_size), Image.Resampling.LANCZOS)
        
        logger.info(f"Image preprocessed to {target_size}x{target_size}")
        return image
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise


def generate_embedding(model, image: Image.Image) -> np.ndarray:
    """
    Generate embedding from Derm Foundation model
    
    Args:
        model: Loaded Derm Foundation model
        image: Preprocessed PIL Image (448x448)
    
    Returns:
        Embedding vector as numpy array
    """
    try:
        # Save image to bytes
        buf = BytesIO()
        image.save(buf, format='JPEG')
        image_bytes = buf.getvalue()
        
        # Format input as TensorFlow Example
        input_tensor = tf.train.Example(features=tf.train.Features(
            feature={'image/encoded': tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[image_bytes]))
            })).SerializeToString()
        
        # Call inference
        infer = model.get_inference_function()
        output = infer(inputs=tf.constant([input_tensor]))
        
        # Extract embedding
        if 'embedding' in output:
            embedding_vector = output['embedding'].numpy().flatten()
        else:
            # Use first available output
            key = list(output.keys())[0]
            embedding_vector = output[key].numpy().flatten()
        
        logger.info(f"Generated embedding of shape {embedding_vector.shape}")
        return embedding_vector
        
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        raise


def validate_image(file_content: bytes, max_size_mb: int = 10) -> bool:
    """
    Validate uploaded image
    
    Args:
        file_content: Raw image bytes
        max_size_mb: Maximum allowed file size in MB
    
    Returns:
        True if valid, False otherwise
    """
    # Check file size
    size_mb = len(file_content) / (1024 * 1024)
    if size_mb > max_size_mb:
        logger.warning(f"Image too large: {size_mb:.2f}MB > {max_size_mb}MB")
        return False
    
    # Try to open as image
    try:
        image = Image.open(BytesIO(file_content))
        # Check if it's a valid image format
        image.verify()
        return True
    except Exception as e:
        logger.warning(f"Invalid image: {e}")
        return False