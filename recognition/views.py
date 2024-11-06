from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('recognition/model/digit_recognition_model.h5')

def preprocess_image(image):
    """Preprocess the uploaded image to match model input format."""
    image = ImageOps.grayscale(image)  # Convert to grayscale
    image = image.resize((28, 28))     # Resize to 28x28 pixels
    image_array = np.array(image) / 255.0  # Normalize pixel values
    return image_array.reshape(1, 28, 28, 1)  # Reshape for model input

def predict_digit(image):
    """Predict the digit in the preprocessed image using the model."""
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return np.argmax(prediction)  # Returns the digit (0–9)

def predict_digits(request):
    prediction = None
    if request.method == 'POST' and request.FILES['digit_image']:
        uploaded_file = request.FILES['digit_image']
        fs = FileSystemStorage()
        filename = fs.save(uploaded_file.name, uploaded_file)
        uploaded_file_path = fs.url(filename)

        # Load the image and make a prediction
        image = Image.open(uploaded_file)
        prediction = predict_digit(image)

    return render(request, 'recognition/index.html', {'prediction': prediction})
