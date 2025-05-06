import numpy as np
from PIL import Image
from . import constants as cs

def preprocess_image(image_path, target_size=cs.UNISHAPE):
    """
    Load and preprocess an external image.
    - Resize the image.
    - Normalize the pixel values.
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size)  # Resize the image
    img_array = np.array(img)  # Convert image to numpy array
    
    # Normalize to [-1, 1] range as expected by most models
    img_array = (img_array / 127.5) - 1.0

    # Ensure the image has the correct shape (batch_size, height, width, channels)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array
