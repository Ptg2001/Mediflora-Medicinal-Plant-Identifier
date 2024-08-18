import requests
from flask import Flask, render_template, request, jsonify, send_file, Response
from fuzzywuzzy import fuzz
from gtts import gTTS
import speech_recognition as sr
import cv2
import numpy as np
import tensorflow as tf
import json

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'

# Load the trained model
model = tf.keras.models.load_model("model/model_sev.h5")
all_plant_names = ['Arive-Dantu', 'Basale', 'Betel', 'Crape_Jasmine', 'Curry', 'Drumstick', 'Fenugreek', 'Guava',
                   'Hibiscus', 'Indian_Beech', 'Indian_Mustard', 'Jackfruit', 'Jamaica', 'Jamun', 'Jasmine',
                   'Karanda', 'Lemon', 'Mango', 'Mexican_Mint', 'Mint', 'Neem', 'Oleander', 'Parijata', 'Peepal',
                   'Pomegranate', 'Rasna', 'Rose_Apple', 'Roxburgh_fig', 'Sandalwood', 'Tulsi']

# Create a voice recognizer
recognizer = sr.Recognizer()

# Variable to track 'p' key press and image capture
capture_image_flag = False

# Load plant information from JSON file
with open('model/plants.json', 'r') as json_file:
    plant_data = json.load(json_file)

# Create a dictionary to store plant descriptions
plant_descriptions = {plant['name']: plant['description'] for plant in plant_data['plants']}

# Function to capture frames from webcam, predict the plant, and overlay the prediction on the frame
def generate_frames():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("Error: Could not open camera.")

    while True:
        success, frame = cap.read()

        if not success:
            break
        else:
            # Preprocess the image
            preprocessed_image = preprocess_image(frame)

            # Predict the plant
            predictions = model.predict(preprocessed_image)
            predicted_class = all_plant_names[predictions.argmax()]

            # Get the plant description based on the predicted class
            description = plant_descriptions.get(predicted_class, "Description not available for this plant.")

            # Display the predicted plant and its description on the frame
            cv2.putText(frame, f"Predicted Plant: {predicted_class}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Description: {description}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Encode the frame to JPEG format
            _, buffer = cv2.imencode('.jpg', frame)

            # Convert the buffer to bytes
            frame_bytes = buffer.tobytes()

            # Yield the frame with appropriate MIME type
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()  # Release the camera when it's no longer needed
    cv2.destroyAllWindows()

# Preprocess the image before prediction
def preprocess_image(frame):
    try:
        # Resize the captured image to the size expected by the model
        input_size = (150, 150)
        resized_frame = cv2.resize(frame, input_size)

        # Normalize pixel values to be in the range [0, 1]
        preprocessed_image = resized_frame.astype(np.float32) / 255.0

        # Ensure color channels are in the correct order (RGB)
        preprocessed_image = preprocessed_image[..., ::-1]

        # Reshape the image to match the model's input shape (add batch dimension)
        preprocessed_image = np.expand_dims(preprocessed_image, axis=0)

        return preprocessed_image

    except Exception as e:
        raise Exception(f"Error preprocessing image: {str(e)}")

# Route for predicting image
@app.route('/predict_image', methods=['POST'])
def predict_image():
    try:
        image = request.files['image']

        # Load and preprocess the image
        img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
        preprocessed_image = preprocess_image(img)

        # Make predictions using the model
        predictions = model.predict(preprocessed_image)
        predicted_class = all_plant_names[predictions.argmax()]

        # Get the plant description based on the predicted class
        description = plant_descriptions.get(predicted_class, "Description not available for this plant.")

        # Return the predicted class and description as JSON
        return jsonify({'predicted_plant': predicted_class, 'description': description})
    except Exception as e:
        return jsonify({'error': str(e)})

# Route for capturing image
@app.route('/capture_image', methods=['POST'])
def capture_image():
    try:
        # Capture an image from the camera
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        ret, frame = cap.read()
        if not ret:
            return jsonify({'message': 'Error capturing image.'})

        # Resize the captured image to the size expected by the model
        input_size = (150, 150)
        img = cv2.resize(frame, input_size)
        img = img.astype(np.float32) / 255.0  # Normalize pixel values
        img = np.expand_dims(img, axis=0)  # Add batch dimension

        # Predict the plant
        predictions = model.predict(img)
        predicted_class = all_plant_names[predictions.argmax()]

        # Get the plant description based on the predicted class
        description = plant_descriptions.get(predicted_class, "Description not available for this plant.")

        cap.release()
        cv2.destroyAllWindows()

        return jsonify({'predicted_plant': predicted_class, 'description': description})

    except Exception as e:
        return jsonify({'message': f'Error: {str(e)}'})

# Route for voice command
@app.route('/voice_command', methods=['POST'])
def voice_command():
    try:
        with sr.Microphone() as source:
            print("Please say something...")
            recognizer.adjust_for_ambient_noise(source, duration=1)

            # Retry capturing audio up to 3 times
            for _ in range(3):
                try:
                    audio = recognizer.listen(source, timeout=10, phrase_time_limit=10)
                    break
                except sr.WaitTimeoutError:
                    print("Retrying due to timeout...")

            # Recognize speech using Google Web Speech API
            query = recognizer.recognize_google(audio, language='en-US', show_all=True)

        if 'alternative' in query:
            recognized_text = query['alternative'][0]['transcript']
            print(f"Recognized Text: {recognized_text}")

            # Find the closest matching plant name
            best_match = max(plant_descriptions.keys(), key=lambda plant_name: fuzz.ratio(recognized_text.lower(), plant_name.lower()))

            # Find the description of the recognized plant
            description = plant_descriptions.get(best_match, "Description not available for this plant.")

            # Convert the description to speech using gTTS
            tts = gTTS(description)
            tts.save("static/plant_description.mp3")

            return jsonify({
                'voice_command_result': f"Predicted Plant: {best_match}\nDescription: {description}",
                'tts_audio_url': '/get_tts_audio',
                'predicted_plant': best_match,
            })
        else:
            return jsonify({'voice_command_result': 'No alternative transcript available.'})
    except sr.WaitTimeoutError:
        print("Speech recognition timed out.")
        return jsonify({'voice_command_result': 'Speech recognition timed out.'})
    except sr.UnknownValueError:
        print("Google Web Speech API could not understand audio.")
        return jsonify({'voice_command_result': 'Could not understand audio.'})
    except sr.RequestError as e:
        print(f"Could not request results from Google Web Speech API; {e}")
        return jsonify({'voice_command_result': f'Error: {e}'})

# Route to get TTS audio
@app.route('/get_tts_audio')
def get_tts_audio():
    return send_file("static/plant_description.mp3", mimetype="audio/mpeg")

# Route for streaming video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
