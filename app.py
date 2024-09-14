import json
import os
from flask import Flask, render_template, request, jsonify, send_file, Response, redirect, url_for, session
from fuzzywuzzy import fuzz
import pyttsx3
import speech_recognition as sr
import tensorflow as tf
import numpy as np
import cv2
from werkzeug.security import generate_password_hash, check_password_hash
from flask_pymongo import PyMongo
from gtts import gTTS
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# MongoDB setup
app.config['MONGO_URI'] = 'mongodb+srv://piyushhole:Piyushhole2001@ecom.neu3z5n.mongodb.net/users?retryWrites=true&w=majority'
app.secret_key = 'secret_key'  # Needed for session management
mongo = PyMongo(app)
users_collection = mongo.db.users

# Load the trained model
model = tf.keras.models.load_model("model/model_sev.h5")
all_plant_names = ['Arive-Dantu', 'Basale', 'Betel', 'Crape_Jasmine', 'Curry', 'Drumstick', 'Fenugreek', 'Guava',
                   'Hibiscus', 'Indian_Beech', 'Indian_Mustard', 'Jackfruit', 'Jamaica', 'Jamun', 'Jasmine',
                   'Karanda', 'Lemon', 'Mango', 'Mexican_Mint', 'Mint', 'Neem', 'Oleander', 'Parijata', 'Peepal',
                   'Pomegranate', 'Rasna', 'Rose_Apple', 'Roxburgh_fig', 'Sandalwood', 'Tulsi']

# Create a voice recognizer
recognizer = sr.Recognizer()

# Load plant information from JSON file
with open('plants.json', 'r', encoding='utf-8') as json_file:
    plant_data = json.load(json_file)

# Create a dictionary to store plant descriptions
plant_descriptions = {'en': {}, 'hi': {}, 'mr': {}}

# Populate plant_descriptions from the loaded JSON data
for plant in plant_data['plants']:
    plant_name = plant.get('name')
    descriptions = plant.get('descriptions', {})

    if plant_name:
        plant_descriptions['en'][plant_name] = descriptions.get('en', 'No description available.')
        plant_descriptions['hi'][plant_name] = descriptions.get('hi', 'No description available.')
        plant_descriptions['mr'][plant_name] = descriptions.get('mr', 'No description available.')

def generate_frames():
    global capture_image_flag
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")

    while True:
        success, frame = cap.read()
        if not success:
            break
        if capture_image_flag:
            # Preprocess the captured image for prediction
            img = cv2.resize(frame, (150, 150))
            img = img / 255.0
            img = img.reshape(1, 150, 150, 3)

            # Predict the plant
            predictions = model.predict(img)
            predicted_class = all_plant_names[predictions.argmax()]

            # Display the predicted plant on the frame
            cv2.putText(frame, f"Predicted Plant: {predicted_class}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()

def preprocess_image(frame):
    try:
        # Resize the captured image to 150x150 pixels
        resized_frame = cv2.resize(frame, (150, 150))

        # Normalize pixel values to be in the range [0, 1]
        preprocessed_image = resized_frame.astype(np.float32) / 255.0

        # Ensure color channels are in the correct order (RGB)
        preprocessed_image = preprocessed_image[..., ::-1]

        # Reshape the image to match the model's input shape (add batch dimension)
        preprocessed_image = np.expand_dims(preprocessed_image, axis=0)

        return preprocessed_image

    except Exception as e:
        raise Exception(f"Error preprocessing image: {str(e)}")

@app.route('/predict_image', methods=['POST'])
def predict_image():
    try:
        image = request.files['image']
        img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
        preprocessed_image = preprocess_image(img)

        # Make predictions using the model
        predictions = model.predict(preprocessed_image)
        predicted_class = all_plant_names[predictions.argmax()]

        # Get the plant description based on the predicted class and selected language
        lang = request.form.get('language', 'en')
        description = plant_descriptions.get(lang, {}).get(predicted_class, "Description not available for this plant.")

        return jsonify({'predicted_plant': predicted_class, 'description': description})
    except Exception as e:
        return jsonify({'error': str(e)})

def generate_tts(description, lang):
    speech = gTTS(text=description, lang=lang, slow=False)
    tts_file_path = 'static/plant_description.mp3'
    speech.save(tts_file_path)
    return tts_file_path

@app.route('/tts', methods=['POST'])
def tts():
    try:
        lang = request.form.get('language', 'en')
        plant_name = request.form.get('plant_name')

        if not plant_name:
            return jsonify({'error': 'Plant name not provided.'})

        description = plant_descriptions.get(lang, {}).get(plant_name, "Description not available.")
        tts_file_path = generate_tts(description, lang)

        if not os.path.exists(tts_file_path) or os.path.getsize(tts_file_path) == 0:
            return jsonify({'error': 'TTS file could not be created.'})

        return jsonify({
            'tts_audio_url': url_for('get_tts_audio')
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/get_tts_audio')
def get_tts_audio():
    return send_file('static/plant_description.mp3', as_attachment=False)

@app.route('/')
def redirect_to_signup():
    return redirect(url_for('signup'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        existing_user = users_collection.find_one({"username": username})
        if existing_user:
            return "User already exists!"
        hashed_password = generate_password_hash(password)
        users_collection.insert_one({'username': username, 'password': hashed_password})
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = users_collection.find_one({'username': username})
        if user and check_password_hash(user['password'], password):
            session['username'] = username
            return redirect(url_for('dashboard'))
        else:
            return "Invalid credentials!"
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html')

@app.route('/guide')
def guide():
    return render_template('guide.html')

@app.route('/profile')
def profile():
    return render_template('profile.html')

@app.route('/api/profile')
def get_profile():
    if 'username' not in session:
        return jsonify({'error': 'User not logged in'}), 401

    user = users_collection.find_one({'username': session['username']})
    if not user:
        return jsonify({'error': 'User not found'}), 404

    profile_data = {
        'username': user.get('username', ''),
        'email': user.get('email', ''),
        'fullName': user.get('fullName', ''),
        'phone': user.get('phone', ''),
        'address': user.get('address', ''),
        'dateOfBirth': user.get('dateOfBirth', ''),
        'profilePic': user.get('profilePic', '')  # URL or path to the profile picture
    }

    return jsonify(profile_data)


@app.route('/detect')
def detect():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/capture_image', methods=['POST'])
def capture_image():
    try:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        if not ret:
            return jsonify({'message': 'Error capturing image.'})
        img = cv2.resize(frame, (150, 150))
        img = img / 255.0
        img = img.reshape(1, 150, 150, 3)
        predictions = model.predict(img)
        predicted_class = all_plant_names[predictions.argmax()]
        cap.release()
        cv2.destroyAllWindows()
        return jsonify({'predicted_plant': predicted_class})
    except Exception as e:
        return jsonify({'message': f'Error: {str(e)}'})

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/voice_command', methods=['POST'])
def voice_command():
    try:
        lang = request.form.get('language', 'en')

        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = recognizer.listen(source)
        query = recognizer.recognize_google(audio, language=lang)

        best_match = None
        best_score = 0
        for plant in plant_data['plants']:
            plant_name = plant['name']
            score = fuzz.ratio(query.lower(), plant_name.lower())
            if score > best_score:
                best_match = plant_name
                best_score = score

        if best_match:
            description = plant_descriptions.get(lang, {}).get(best_match, 'Description not available for this plant.')
            tts_file_path = generate_tts(description, lang)
            return jsonify({'plant_name': best_match, 'description': description, 'tts_audio_url': url_for('get_tts_audio')})
        else:
            return jsonify({'error': 'No matching plant found.'})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
