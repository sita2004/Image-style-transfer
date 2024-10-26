from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import io
import base64

app = FastAPI()

# Function to load TensorFlow Hub model only when needed
def load_model():
    return hub.load("https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2")

# Function to process and load image data for model input
def load_and_process_image(image_data, image_size=(256, 256)):
    img = Image.open(io.BytesIO(image_data))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = tf.convert_to_tensor(np.array(img))
    img = tf.image.resize(img, image_size)
    img = tf.expand_dims(img, 0) / 255.0  # Normalize to [0, 1]
    return img

@app.get("/")
async def home():
    return {"message": "Welcome to the Image Style Transfer API!"}

@app.post("/stylize")
async def stylize(content_image: UploadFile = File(...), style_image: UploadFile = File(...)):
    try:
        # Read images from the request
        content_image_data = await content_image.read()
        style_image_data = await style_image.read()

        # Check if both images are provided
        if not content_image_data or not style_image_data:
            raise HTTPException(status_code=400, detail="Both content_image and style_image must be provided.")

        # Load and process images for model input
        content_image_tensor = load_and_process_image(content_image_data)
        style_image_tensor = load_and_process_image(style_image_data)

        # Load model and apply style transfer
        hub_module = load_model()
        stylized_image = hub_module(tf.constant(content_image_tensor), tf.constant(style_image_tensor))[0]

        # Convert tensor back to image format
        stylized_image = np.squeeze(stylized_image) * 255
        stylized_image = Image.fromarray(np.uint8(stylized_image))

        # Convert to base64 for JSON response
        buf = io.BytesIO()
        stylized_image.save(buf, format="PNG")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        # Send JSON response with base64-encoded image
        return JSONResponse(content={"stylized_image": img_base64})

    except HTTPException as e:
        raise e
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
