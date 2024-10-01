from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, session
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
import time
import os
from bson import ObjectId
import re

# Initialize Flask app
app = Flask(__name__)

# MongoDB setup
app.config['MONGO_URI'] = 'mongodb+srv://piyushhole:Piyushhole2001@ecom.neu3z5n.mongodb.net/users?retryWrites=true&w=majority'
app.secret_key = 'secret_key'  # Needed for session management
mongo = PyMongo(app)
users_collection = mongo.db.users
plants_collection = mongo.db.plants  # Access the 'plants' collection
plantslist_collection = mongo.db.plantslist

# Load the trained model
model_path = os.path.join(os.getcwd(), 'model', 'model_sev.h5')
model = tf.keras.models.load_model(model_path)

all_plant_names = ['Arive-Dantu', 'Basale', 'Betel', 'Crape_Jasmine', 'Curry', 'Drumstick', 'Fenugreek', 'Guava',
                   'Hibiscus', 'Indian_Beech', 'Indian_Mustard', 'Jackfruit', 'Jamaica', 'Jamun', 'Jasmine',
                   'Karanda', 'Lemon', 'Mango', 'Mexican_Mint', 'Mint', 'Neem', 'Oleander', 'Parijata', 'Peepal',
                   'Pomegranate', 'Rasna', 'Rose_Apple', 'Roxburgh_fig', 'Sandalwood', 'Tulsi']

# Create a voice recognizer
recognizer = sr.Recognizer()

# Fetch plant information from MongoDB
def load_plant_descriptions():
    plant_descriptions = {'en': {}, 'hi': {}, 'mr': {}}
    plants = plants_collection.find()

    for plant in plants:
        plant_name = plant.get('name')
        descriptions = plant.get('descriptions', {})
        if plant_name:
            plant_descriptions['en'][plant_name] = descriptions.get('en', 'No description available.')
            plant_descriptions['hi'][plant_name] = descriptions.get('hi', 'No description available.')
            plant_descriptions['mr'][plant_name] = descriptions.get('mr', 'No description available.')
    return plant_descriptions

# Initialize plant descriptions from MongoDB
plant_descriptions = load_plant_descriptions()

def preprocess_image(frame):
    try:
        resized_frame = cv2.resize(frame, (150, 150))
        preprocessed_image = resized_frame.astype(np.float32) / 255.0
        preprocessed_image = preprocessed_image[..., ::-1]
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
        predictions = model.predict(preprocessed_image)
        predicted_class = all_plant_names[predictions.argmax()]
        lang = request.form.get('language', 'en')
        description = plant_descriptions.get(lang, {}).get(predicted_class, "Description not available for this plant.")
        return jsonify({'predicted_plant': predicted_class, 'description': description})
    except Exception as e:
        return jsonify({'error': str(e)})

def cleanup_old_files():
    try:
        directory = 'static'
        now = time.time()
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if filename.endswith('.mp3') and os.path.isfile(file_path):
                if now - os.path.getmtime(file_path) > 10:
                    os.remove(file_path)
                    print(f"Deleted old file: {filename}")
    except Exception as e:
        print(f"Error during cleanup: {str(e)}")

def generate_tts(description, lang):
    cleanup_old_files()
    timestamp = str(int(time.time()))
    tts_file_path = f'static/plant_description_{timestamp}.mp3'
    speech = gTTS(text=description, lang=lang, slow=False)
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
        return jsonify({'tts_audio_url': url_for('get_tts_audio', filename=os.path.basename(tts_file_path))})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/get_tts_audio/<filename>')
def get_tts_audio(filename):
    file_path = os.path.join('static', filename)
    return send_file(file_path, as_attachment=False)

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

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))

    try:
        user = users_collection.find_one({'username': session['username']})
        plants = list(plants_collection.find())
        plantslist = list(plantslist_collection.find())
        total_users = users_collection.count_documents({})
        total_plants = len(plants)
        total_plantslist = len(plantslist)

        return render_template('dashboard.html', user=user, plants=plants, plantslist=plantslist,
                               total_users=total_users, total_plants=total_plants,
                               total_plantslist=total_plantslist)
    except Exception as e:
        print(f"Error fetching data: {e}")
        return render_template('error.html', message="Failed to fetch dashboard data.")

@app.route('/guide')
def guide():
    return render_template('guide.html')

@app.route('/profile')
def profile():
    return render_template('profile.html')

@app.route('/plantlist')
def plantlist():
    return render_template('plantlist.html')

@app.route('/api/plants', methods=['GET'])
def get_plants():
    plants = []
    for plant in plantslist_collection.find():
        plants.append({
            'id': str(plant['_id']),
            'name': plant.get('name'),
            'description': plant.get('description', 'No description available.'),
            'leafImages': plant.get('leafImages', []),
            'regions': plant.get('regions', []),  # Changed to 'regions' for consistency
            'wikipediaLink': plant.get('wikipediaLink', 'No link available.'),
            'locations': [loc['coordinates'] for loc in plant.get('locations', [])]  # Flatten locations
        })
    return jsonify(plants)

@app.route('/api/plants/<string:plant_id>', methods=['GET'])
def get_plant(plant_id):
    if not re.match(r'^[0-9a-f]{24}$', plant_id):
        return jsonify({'error': 'Invalid plant ID format'}), 400

    plant = plantslist_collection.find_one({"_id": ObjectId(plant_id)})

    if plant:
        return jsonify({
            'id': str(plant['_id']),
            'name': plant.get('name'),
            'description': plant.get('description', 'No description available.'),
            'leafImages': plant.get('leafImages', []),
            'regions': plant.get('regions', []),
            'wikipediaLink': plant.get('wikipediaLink', 'No link available.'),
            'locations': [loc['coordinates'] for loc in plant.get('locations', [])]
        })
    else:
        return jsonify({'error': 'Plant not found'}), 404

@app.route('/api/plants/<string:plant_id>/delete', methods=['DELETE'])
def delete_plant(plant_id):
    if not re.match(r'^[0-9a-f]{24}$', plant_id):
        return jsonify({'error': 'Invalid plant ID format'}), 400

    result = plantslist_collection.delete_one({"_id": ObjectId(plant_id)})

    if result.deleted_count == 1:
        return jsonify({'message': 'Plant deleted successfully'})
    else:
        return jsonify({'error': 'Plant not found'}), 404
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
            return jsonify({'message': 'Error capturing image'}), 500

        file_name = 'captured_image.jpg'
        file_path = os.path.join('static', file_name)
        cv2.imwrite(file_path, frame)
        cap.release()

        return jsonify({'image_url': url_for('static', filename=file_name)})
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/recognize', methods=['POST'])
def recognize():
    try:
        # Retrieve the captured image
        image_url = request.form.get('image_url')
        file_path = os.path.join('static', image_url.split('/')[-1])

        # Read and preprocess the image
        img = cv2.imread(file_path)
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

if __name__ == '__main__':
    app.run(debug=True)
