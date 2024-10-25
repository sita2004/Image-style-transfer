from flask import Flask, request, jsonify
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from flask_cors import CORS
from PIL import Image
import io
import base64
import os

app = Flask(__name__)
CORS(app)

# Load the TensorFlow Hub model
hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
hub_module = hub.load(hub_handle)

def load_and_process_image(image_data, image_size=(256, 256)):
    # Open and process the image
    img = Image.open(io.BytesIO(image_data))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    img = tf.convert_to_tensor(np.array(img))
    img = tf.image.resize(img, image_size)
    img = tf.expand_dims(img, 0) / 255.0
    return img

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Welcome to the Image Style Transfer API!"})

@app.route('/stylize', methods=['POST'])
def stylize():
    content_image_data = request.files['content_image'].read()
    style_image_data = request.files['style_image'].read()

    # Process content and style images
    content_image = load_and_process_image(content_image_data)
    style_image = load_and_process_image(style_image_data)

    # Perform stylization
    stylized_image = hub_module(tf.constant(content_image), tf.constant(style_image))[0]

    # Convert tensor back to image format
    stylized_image = np.squeeze(stylized_image) * 255
    stylized_image = Image.fromarray(np.uint8(stylized_image))

    # Encode image in base64
    buf = io.BytesIO()
    stylized_image.save(buf, format='PNG')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    # Return JSON with base64-encoded image
    return jsonify({'stylized_image': img_base64})

# Set the app to run on the assigned port in deployment
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)