from flask import Flask, jsonify, request, render_template
import tensorflow as tf
import numpy as np
import cv2

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('covid_model.h5')

# Define a function to preprocess the image
def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Define the home page route
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if the file key is present in the request object
    if 'file' not in request.files:
        return jsonify({'error': 'no file uploaded'})

    # Get the file from the request
    file = request.files['file']

    # Read the file and preprocess the image
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    image = preprocess_image(image)

    # Make a prediction using the model
    prediction = model.predict(image)[0][0]

    # Format the prediction result as a JSON object
    result = {'prediction': float(prediction)}

    # Return the prediction result as a JSON response
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)
