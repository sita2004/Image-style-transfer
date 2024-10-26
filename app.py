from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import io
import base64
import os

# Disable GPU and suppress TensorFlow non-error logs
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable CUDA (GPU)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Only log errors

# Disable GPU initialization at TensorFlow level
tf.config.set_visible_devices([], 'GPU')

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust allowed origins as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the TensorFlow Hub model
hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
hub_module = hub.load(hub_handle)

def load_and_process_image(image_data, image_size=(256, 256)):
    # Open and process the image
    img = Image.open(io.BytesIO(image_data))
    
    # Ensure the image is in RGB format
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Convert to TensorFlow tensor, resize, and normalize
    img = tf.convert_to_tensor(np.array(img))
    img = tf.image.resize(img, image_size)
    img = tf.expand_dims(img, 0) / 255.0  # Normalize to [0, 1]
    return img

@app.get("/")
async def home():
    return JSONResponse(content={"message": "Welcome to the Image Style Transfer API!"})

@app.post("/stylize")
async def stylize(content_image: UploadFile = File(...), style_image: UploadFile = File(...)):
    try:
        # Read content and style images
        content_image_data = await content_image.read()
        style_image_data = await style_image.read()

        # Process content and style images
        content_image_tensor = load_and_process_image(content_image_data)
        style_image_tensor = load_and_process_image(style_image_data)

        # Perform stylization
        stylized_image = hub_module(tf.constant(content_image_tensor), tf.constant(style_image_tensor))[0]

        # Convert tensor back to image format
        stylized_image = np.squeeze(stylized_image) * 255
        stylized_image = Image.fromarray(np.uint8(stylized_image))

        # Encode image in base64
        buf = io.BytesIO()
        stylized_image.save(buf, format='PNG')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        # Return JSON with base64-encoded image
        return JSONResponse(content={'stylized_image': img_base64})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))  # Handle unexpected errors

# Run the FastAPI application with Uvicorn for deployment
# In a production environment, run using: `uvicorn app:app --host 0.0.0.0 --port 5000`
