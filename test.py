from flask import Flask, request
from flask_cors import CORS
from flask import send_from_directory
from Mlops.pipeline.prediction import PredictionPipeline

app = Flask(__name__)
CORS(app)


@app.route('/')
def index():
    return send_from_directory('templates', 'index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No image provided"

    image_file = request.files['image']
    if image_file.filename == '':
        return "No selected file"

    image_path = "uploaded_image.jpg"  # Save the uploaded image temporarily
    image_file.save(image_path)

    predictor = PredictionPipeline(image_path)
    prediction = predictor.predict()

    return prediction  # Return the prediction result as plain text


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
