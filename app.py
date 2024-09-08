from flask import Flask, request, render_template, send_file
import os
import cv2
import numpy as np
from gfpgan_inference import enhance_images

app = Flask(__name__)

# Save uploaded images in this folder temporarily
UPLOAD_FOLDER = 'static/uploads/'
RESULT_FOLDER = 'static/results/'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/enhance-image/', methods=['POST'])
def enhance_image_endpoint():
    if 'image' not in request.files:
        return "No file part", 400

    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400

    # Save the uploaded image temporarily
    img_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(img_path)

    # Load the image with OpenCV
    input_img = cv2.imread(img_path)

    # Enhance the image using GFPGAN
    cropped_faces, restored_faces, restored_img = enhance_images(
        input_img,
        version=request.form.get('version', '1.3'),
        upscale=int(request.form.get('upscale', 2)),
        bg_upsampler=request.form.get('bg_upsampler', 'realesrgan'),
        only_center_face=bool(request.form.get('only_center_face', False)),
        aligned=bool(request.form.get('aligned', False))
    )

    # Save the enhanced image
    enhanced_img_path = os.path.join(RESULT_FOLDER, file.filename)
    if restored_img is not None:
        cv2.imwrite(enhanced_img_path, restored_img)

    # Return the enhanced image as a response
    return send_file(enhanced_img_path, mimetype='image/jpeg')


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
