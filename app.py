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
import uvicorn
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)

app = FastAPI()

# Add CORS middleware to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://image-style-transfer-frontend.onrender.com"],  # Update with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the TensorFlow Hub model once, with logging
logging.info("Loading the TensorFlow Hub model...")
hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
hub_module = hub.load(hub_handle)
logging.info("Model loaded successfully.")

# Function to load and process image data
def load_and_process_image(image_data, image_size=(256, 256)):
    img = Image.open(io.BytesIO(image_data))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(image_size, Image.LANCZOS)  # Updated to use Image.LANCZOS
    img = np.array(img) / 255.0  # Normalize to [0, 1]
    img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
    return tf.expand_dims(img_tensor, 0)

@app.get("/", include_in_schema=False)
async def home():
    return {"message": "Welcome to the Image Style Transfer API!"}

@app.post("/stylize")
async def stylize(content_image: UploadFile = File(...), style_image: UploadFile = File(...)):
    try:
        # Read and validate images from request
        content_image_data = await content_image.read()
        style_image_data = await style_image.read()
        if not content_image_data or not style_image_data:
            raise HTTPException(status_code=400, detail="Both content_image and style_image must be provided.")

        # Process images for model input
        content_image_tensor = load_and_process_image(content_image_data)
        style_image_tensor = load_and_process_image(style_image_data)

        # Apply style transfer
        logging.info("Applying style transfer...")
        stylized_image = hub_module(tf.constant(content_image_tensor), tf.constant(style_image_tensor))[0]

        # Convert tensor back to image format
        stylized_image = np.squeeze(stylized_image) * 255
        stylized_image = Image.fromarray(np.uint8(stylized_image))

        # Convert image to base64 format
        buf = io.BytesIO()
        stylized_image.save(buf, format="PNG")
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        # Return the base64-encoded image in JSON response
        logging.info("Style transfer successful, returning response.")
        return JSONResponse(content={"stylized_image": img_base64})

    except HTTPException as e:
        raise e
    except Exception as e:
        logging.error(f"Error during style transfer: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    # Bind to the PORT environment variable or use default port 8000
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, timeout_keep_alive=300)  # Extended timeout
