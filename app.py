from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import io

app = FastAPI()

# Load the TensorFlow Hub model
hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
hub_module = hub.load(hub_handle)

# Process and load image data
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
        # Read images from request
        content_image_data = await content_image.read()
        style_image_data = await style_image.read()

        # Check if both images are provided
        if not content_image_data or not style_image_data:
            raise HTTPException(status_code=400, detail="Both content_image and style_image must be provided.")

        # Process images for model input
        content_image_tensor = load_and_process_image(content_image_data)
        style_image_tensor = load_and_process_image(style_image_data)

        # Apply style transfer
        stylized_image = hub_module(tf.constant(content_image_tensor), tf.constant(style_image_tensor))[0]

        # Convert tensor back to image format
        stylized_image = np.squeeze(stylized_image) * 255
        stylized_image = Image.fromarray(np.uint8(stylized_image))

        # Save image to a byte buffer
        buf = io.BytesIO()
        stylized_image.save(buf, format="PNG")
        buf.seek(0)

        # Return the image as a streaming response
        return StreamingResponse(buf, media_type="image/png")

    except HTTPException as e:
        raise e
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
