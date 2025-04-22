from flask import Flask, render_template, request, redirect, url_for
from PIL import Image, ImageFilter
import numpy as np
import os
import matplotlib.pyplot as plt
import google.generativeai as genai
import io
import base64
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'

# Check if the path is a file or directory before creating the directory
if os.path.exists(UPLOAD_FOLDER):
    if os.path.isfile(UPLOAD_FOLDER):
        os.remove(UPLOAD_FOLDER)  # Remove the file if it exists
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create the directory
    elif not os.path.isdir(UPLOAD_FOLDER):
        raise FileExistsError(f"A file with the name '{UPLOAD_FOLDER}' already exists and cannot be handled.")
else:
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Gemini API setup
genai.configure(api_key="AIzaSyBaczF7Z18gzWXddh9NicOtmfln8oEA2A8")
model = genai.GenerativeModel("gemini-2.0-flash")

def analyze_image_with_gemini(img, title):
    try:
        response = model.generate_content([
            f"This is an astronomical image processed with {title}. Describe the color distribution, temperature difference, what it signify etc in 3 different lines. Don't give any extra text...just to the point.",img  
        ])
        return response.text.strip()
    except Exception as e:
        return f"Error: {e}"

def image_to_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            print("✅ Saved at:", filepath)
            image = Image.open(filepath).convert("RGB")

            # Convert image to numpy array for processing
            img_array = np.array(image)
            gray_array = np.array(image.convert("L"))

            results = []

            # 1. Original
            results.append({
                'title': 'Original Image',
                'image': image_to_base64(image),
                'ai': analyze_image_with_gemini(image, "Original Image")
            })

            # 2. Contrast Stretching
            contrast_img = Image.fromarray(np.interp(img_array, (img_array.min(), img_array.max()), (0, 255)).astype(np.uint8))
            results.append({
                'title': 'Contrast Stretching',
                'image': image_to_base64(contrast_img),
                'ai': analyze_image_with_gemini(contrast_img, "Contrast Stretching")
            })

            # 3. Negative
            negative_img = Image.fromarray(255 - img_array)
            results.append({
                'title': 'Negative Image',
                'image': image_to_base64(negative_img),
                'ai': analyze_image_with_gemini(negative_img, "Negative Image")
            })

            # 4. Log Transform
            log_img_array = np.log1p(img_array)
            log_img = Image.fromarray(np.uint8(255 * log_img_array / np.max(log_img_array)))
            results.append({
                'title': 'Log Transformation',
                'image': image_to_base64(log_img),
                'ai': analyze_image_with_gemini(log_img, "Log Transformation")
            })

            # 5. Gamma
            gamma = 1.5
            gamma_img_array = 255 * (img_array / 255) ** gamma
            gamma_img = Image.fromarray(np.uint8(gamma_img_array))
            results.append({
                'title': 'Gamma Transformation',
                'image': image_to_base64(gamma_img),
                'ai': analyze_image_with_gemini(gamma_img, "Gamma Transformation")
            })

            # 6. Histogram Equalization
            hist, bins = np.histogram(gray_array.flatten(), 256, [0, 256])
            cdf = hist.cumsum()
            cdf_normalized = 255 * cdf / cdf[-1]
            hist_eq_array = cdf_normalized[gray_array.astype(int)]
            hist_eq_img = Image.fromarray(np.uint8(hist_eq_array))
            results.append({
                'title': 'Histogram Equalized',
                'image': image_to_base64(hist_eq_img),
                'ai': analyze_image_with_gemini(hist_eq_img, "Histogram Equalized")
            })

            # 7. Smoothing
            smoothed_img = image.filter(ImageFilter.BoxBlur(2))
            results.append({
                'title': 'Smoothed Image',
                'image': image_to_base64(smoothed_img),
                'ai': analyze_image_with_gemini(smoothed_img, "Smoothed Image")
            })

            # 8. Sharpening
            laplacian_kernel = ImageFilter.Kernel((3, 3), [-1, -1, -1, -1, 8, -1, -1, -1, -1], 1, 0)
            sharpened_img = image.filter(laplacian_kernel)
            results.append({
                'title': 'Sharpened Image (Laplacian)',
                'image': image_to_base64(sharpened_img),
                'ai': analyze_image_with_gemini(sharpened_img, "Sharpened Image")
            })

            # 9. Thresholding
            threshold_img = image.convert("L").point(lambda p: 255 if p > 120 else 0)
            results.append({
                'title': 'Thresholded Image',
                'image': image_to_base64(threshold_img),
                'ai': analyze_image_with_gemini(threshold_img, "Thresholded Image")
            })

            return render_template("index.html", results=results)

    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)
