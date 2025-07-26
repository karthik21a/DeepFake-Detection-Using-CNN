import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('deepfake_cnn_model.keras')

# Define the directory to save uploaded images
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Prediction function
def predict_image(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize the image

    # Predict using the model
    predictions = model.predict(img_array)

    # Adjust this based on your class mapping (fake:0, real:1)
    if predictions[0] > 0.5:
        return "Real"
    else:
        return "Fake"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']
    
    if file.filename == '':
        return 'No selected file'
    
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Get prediction
        result = predict_image(file_path)
        
        # Return the result to the user
        return render_template('result.html', result=result, image_path=file_path)

if __name__ == '__main__':
    app.run(debug=True)
