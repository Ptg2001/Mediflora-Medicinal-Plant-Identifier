from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, session
from rapidfuzz import fuzz
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
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
import json
from datetime import datetime
import requests
from io import BytesIO
import torch
import torchvision.transforms as transforms
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor
import base64
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# Initialize Flask app
app = Flask(__name__)

# MongoDB setup
app.config['MONGO_URI'] = 'mongodb+srv://piyushhole:Piyushhole2001@ecom.neu3z5n.mongodb.net/users?retryWrites=true&w=majority'
app.secret_key = 'secret_key'  # Needed for session management
mongo = PyMongo(app)
users_collection = mongo.db.users
plants_collection = mongo.db.plants  # Access the 'plants' collection
plantslist_collection = mongo.db.plantslist
reports_collection = mongo.db.reports

# Load the trained model directly from Hugging Face when needed
MODEL_URL = "https://huggingface.co/ptg2001/mediflora/resolve/main/model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 52

# Image processing
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

all_plant_names = ["Coriender", "Geranium", "Tulasi", "Nooni", "Ekka", "Jackfruit", "Doddapatre", "Pomegranate", "Honge", "Papaya", "Hibiscus", "Sapota", "Tamarind", "Lemon_grass", "Ashoka", "Ashwagandha", "Curry_Leaf", "Curry", "Doddpathre", "Neem", "Pepper", "Aloevera", "Bamboo", "Catharanthus", "Amruta_Balli", "Betel_Nut", "Mint", "Lemon", "Brahmi", "Rose", "Raktachandini", "Insulin", "Avacado", "Tulsi", "Pappaya", "Basale", "Guava", "Henna", "Ganike", "Wood_sorel", "Jasmine", "Seethapala", "Gauva", "Nagadali", "Mango", "Palak(Spinach)", "Arali", "Castor", "Betel", "Nithyapushpa", "Amla", "Bhrami"]

# Lazy load model only when accessing the detect page
def get_model():
    global model
    if 'model' not in globals():
        print("Loading model from Hugging Face...")
        model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", num_labels=num_classes)
        model.load_state_dict(torch.hub.load_state_dict_from_url(MODEL_URL, map_location=device))
        model.to(device)
        model.eval()
    return model

# Image processing
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

all_plant_names = ["Coriender", "Geranium", "Tulasi", "Nooni", "Ekka", "Jackfruit", "Doddapatre", "Pomegranate",
    "Honge", "Papaya", "Hibiscus", "Sapota", "Tamarind", "Lemon_grass", "Ashoka", "Ashwagandha",
    "Curry_Leaf", "Curry", "Doddpathre", "Neem", "Pepper", "Aloevera", "Bamboo", "Catharanthus",
    "Amruta_Balli", "Betel_Nut", "Mint", "Lemon", "Brahmi", "Rose", "Raktachandini", "Insulin",
    "Avacado", "Tulsi", "Pappaya", "Basale", "Guava", "Henna", "Ganike", "Wood_sorel", "Jasmine",
    "Seethapala", "Gauva", "Nagadali", "Mango", "Palak(Spinach)", "Arali", "Castor", "Betel",
    "Nithyapushpa", "Amla", "Bhrami"]

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

def preprocess_image(image):
    inputs = processor(images=image, return_tensors="pt")
    return inputs["pixel_values"].to(device)

def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = preprocess_image(image)
    with torch.no_grad():
        outputs = model(inputs)
    logits = outputs.logits
    predicted_class = logits.argmax(-1).item()
    return all_plant_names[predicted_class] if predicted_class < len(all_plant_names) else "No such plant found"

@app.route('/predict_image', methods=['POST'])
def predict_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        image = request.files['image']
        image_path = "static/uploaded_image.jpg"
        image.save(image_path)

        predicted_class = predict(image_path)
        
        # Get the description based on selected language
        lang = request.form.get('language', 'en')
        description = plant_descriptions.get(lang, {}).get(predicted_class, "Description not available.")
        
        return jsonify({
            'predicted_plant': predicted_class,
            'description': description
        })
    except Exception as e:
        return jsonify({'error': str(e)})
    
@app.route('/generate_report', methods=['POST'])
def generate_report():
    try:
        plant_name = request.form.get('plant_name')
        if not plant_name:
            return jsonify({'error': 'Plant name not provided'}), 400

        # Fetch plant data from MongoDB
        plant_data = mongo.db.reports.find_one({"name": plant_name})
        if not plant_data:
            return jsonify({'error': 'Plant data not found'}), 404

        # Generate PDF report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"plant_report_{timestamp}.pdf"
        filepath = os.path.join('static', 'reports', filename)

        # Ensure reports directory exists
        os.makedirs(os.path.join('static', 'reports'), exist_ok=True)

        # Create PDF with custom page size and margins
        doc = SimpleDocTemplate(
            filepath,
            pagesize=letter,
            rightMargin=50,
            leftMargin=50,
            topMargin=50,
            bottomMargin=50
        )

        # Define styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=28,
            spaceAfter=30,
            textColor=HexColor('#1a5f7a'),
            alignment=1,
            fontName='Helvetica-Bold'
        )

        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=18,
            spaceBefore=20,
            spaceAfter=12,
            textColor=HexColor('#2c3e50'),
            fontName='Helvetica-Bold'
        )

        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=12,
            textColor=HexColor('#333333'),
            spaceAfter=8,
            fontName='Helvetica'
        )

        list_style = ParagraphStyle(
            'CustomList',
            parent=styles['Normal'],
            fontSize=12,
            leftIndent=20,
            spaceAfter=6,
            bulletIndent=10,
            textColor=HexColor('#333333'),
            fontName='Helvetica'
        )

        story = []

        # Add report title
        story.append(Paragraph(f"Detailed Plant Report", title_style))
        story.append(Paragraph(f"{plant_data['name']} ({plant_data['scientificName']})", heading_style))
        story.append(Spacer(1, 20))

        # Handle images with improved error handling and layout
        if plant_data.get('images'):
            story.append(Paragraph("Plant Images", heading_style))
            story.append(Spacer(1, 10))

            # Process images in pairs
            images = []
            for category, urls in plant_data['images'].items():
                if urls and isinstance(urls, list) and urls[0]:
                    try:
                        response = requests.get(urls[0], timeout=10)
                        if response.status_code == 200:
                            img_data = BytesIO(response.content)
                            img = Image(img_data)
                            # Scale image maintaining aspect ratio
                            aspect = img.imageWidth / float(img.imageHeight)
                            if aspect > 1:
                                img.drawWidth = 3 * inch
                                img.drawHeight = (3 * inch) / aspect
                            else:
                                img.drawHeight = 3 * inch
                                img.drawWidth = 3 * inch * aspect
                            
                            images.append((img, category))
                    except Exception as e:
                        print(f"Error processing image for {category}: {str(e)}")
                        continue

            # Create image tables in pairs
            for i in range(0, len(images), 2):
                image_row = []
                for j in range(2):
                    if i + j < len(images):
                        img, category = images[i + j]
                        image_row.extend([
                            [img],
                            [Paragraph(f"{category.title()}", normal_style)]
                        ])

                if image_row:
                    table = Table([image_row], colWidths=[3*inch] * (len(image_row)//2))
                    table.setStyle(TableStyle([
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                        ('TOPPADDING', (0, 0), (-1, -1), 15),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 15),
                    ]))
                    story.append(table)
                    story.append(Spacer(1, 20))

        # Add basic information
        story.append(Paragraph("Basic Information", heading_style))
        basic_info = [
            ["Scientific Name:", plant_data.get('scientificName', 'N/A')],
            ["Family:", plant_data.get('family', 'N/A')],
            ["Kingdom:", plant_data.get('kingdom', 'N/A')],
            ["Conservation Status:", plant_data.get('conservationStatus', 'N/A')],
            ["Category:", plant_data.get('category', 'N/A')]
        ]

        table = Table(basic_info, colWidths=[2.5*inch, 4*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), HexColor('#f5f9fa')),
            ('TEXTCOLOR', (0, 0), (-1, -1), HexColor('#333333')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('TOPPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, HexColor('#e1e8ed')),
            ('ROWBACKGROUNDS', (0, 0), (-1, -1), [HexColor('#ffffff'), HexColor('#f8f9fa')]),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ]))
        story.append(table)
        story.append(Spacer(1, 20))

        # Add remaining sections (medicinal properties, traditional uses, etc.)
        sections = [
            ('medicinalProperties', "Medicinal Properties", "This plant has been traditionally used for the following medicinal purposes:"),
            ('traditionalUses', "Traditional Uses", "Historical and cultural applications include:"),
            ('chemicalCompounds', "Chemical Compounds", "Key active compounds identified in this plant:"),
            ('habitat', "Natural Habitat", None),
            ('researchPapers', "Scientific Research", "Notable research papers and studies:")
        ]

        for section_key, section_title, section_intro in sections:
            if plant_data.get(section_key):
                story.append(Paragraph(section_title, heading_style))
                
                if section_intro:
                    story.append(Paragraph(section_intro, normal_style))

                if section_key == 'habitat':
                    story.append(Paragraph(plant_data[section_key], normal_style))
                elif section_key == 'researchPapers':
                    for paper in plant_data[section_key]:
                        paper_text = f"{paper['title']} ({paper['year']}) - {', '.join(paper['authors'])}"
                        story.append(Paragraph(f"• {paper_text}", list_style))
                else:
                    for item in plant_data[section_key]:
                        story.append(Paragraph(f"• {item}", list_style))
                
                story.append(Spacer(1, 15))

        # Add footer
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=8,
            textColor=HexColor('#666666'),
            alignment=1
        )
        story.append(Spacer(1, 30))
        story.append(Paragraph(f"Report generated on {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}", footer_style))

        # Build the PDF
        doc.build(story)

        return jsonify({
            'success': True,
            'report_url': url_for('static', filename=f'reports/{filename}')
        })

    except Exception as e:
        print(f"Error generating report: {str(e)}")
        return jsonify({'error': str(e)}), 500


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
    # Get the category from query parameters (if provided)
    category = request.args.get('category')

    # Build the query based on the category
    query = {}
    if category:
        query['category'] = category

    # Fetch plants based on the query
    plants = []
    for plant in plantslist_collection.find(query):  # Use the query to filter by category
        plants.append({
            'id': str(plant['_id']),
            'name': plant.get('name'),
            'description': plant.get('description', 'No description available.'),
            'leafImages': plant.get('leafImages', []),
            'regions': plant.get('regions', []),  # Consistency with regions field
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
    
@app.route('/detect', methods=["POST"])
def detect():
    if 'username' not in session:
        return redirect(url_for('login'))

    # Load the model only when an image is uploaded
    model = get_model()

    image_file = request.files.get("image")
    if not image_file:
        return jsonify({"error": "No image uploaded"}), 400

    return jsonify({"message": "Model loaded and image received!"})



@app.route('/capture_image', methods=['POST'])
def capture_image():
    try:
        # Get the JSON data from the request
        data = request.get_json()
        
        # Check if 'image' is in the data
        if 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400

        # Get the base64 image data
        image_data = data['image']
        
        # Split the header and the base64 string
        header, encoded = image_data.split(',', 1)
        image_bytes = base64.b64decode(encoded)

        # Convert bytes to a numpy array
        np_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        # Save the image to a file (optional, for debugging or further processing)
        image_path = "static/uploaded_image.jpg"
        cv2.imwrite(image_path, image)

        # Perform your prediction logic here
        predicted_class = predict(image_path)  # Replace with your actual prediction function

        return jsonify({'predicted_plant': predicted_class})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_base64', methods=['POST'])
def predict_from_base64():
    try:
        # Get the JSON data from the request
        data = request.get_json()
        
        # Check if 'image' is in the data
        if 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400

        # Get the base64 image data
        image_data = data['image']
        
        # Split the header and the base64 string
        header, encoded = image_data.split(',', 1)
        image_bytes = base64.b64decode(encoded)

        # Convert bytes to a numpy array
        np_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        # Save the image to a file (optional, for debugging or further processing)
        image_path = "static/uploaded_image.jpg"
        cv2.imwrite(image_path, image)

        # Perform your prediction logic here
        predicted_class = predict(image_path)  # Replace with your actual prediction function

        return jsonify({'predicted_plant': predicted_class})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/capture')
def capture():
    return render_template('capture.html')

@app.route('/capture_and_predict', methods=['POST'])
def capture_and_predict():
    # This route will be called to open the camera interface
    return jsonify({'redirect': url_for('capture')})

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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  # Default to 8080 (Render standard)
    print(f"Running on port {port}...")  # Debugging output
    app.run(host="0.0.0.0", port=port)

