import streamlit as st
import tensorflow as tf
import numpy as np
import datetime
import os
import gdown
from googletrans import Translator

# Initialize translator for multilingual support
translator = Translator()

# Define a dictionary of disease-treatment mappings
treatment_recommendations = {
    # Apple
    "Apple___Apple_scab": "Apply fungicides like captan or myclobutanil. Prune and destroy infected leaves. Promote good air circulation.",
    "Apple___Black_rot": "Use fungicides containing copper or captan. Remove and destroy infected fruit and branches. Keep trees well-pruned.",
    "Apple___Cedar_apple_rust": "Apply fungicides with myclobutanil or sulfur. Remove nearby cedar trees if possible. Prune infected areas.",
    "Apple___healthy": "Your apple tree is healthy. No treatment needed.",
    
    # Blueberry
    "Blueberry___healthy": "Your blueberry plant is healthy. No treatment needed.",
    
    # Cherry
    "Cherry_(including_sour)___Powdery_mildew": "Use sulfur-based fungicides or neem oil. Ensure good air circulation around trees. Prune infected branches.",
    "Cherry_(including_sour)___healthy": "Your cherry tree is healthy. No treatment needed.",
    
    # Corn
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Apply fungicides like mancozeb or azoxystrobin. Rotate crops and avoid overhead watering.",
    "Corn_(maize)___Common_rust_": "Use fungicides with active ingredients like chlorothalonil or mancozeb. Plant resistant corn varieties.",
    "Corn_(maize)___Northern_Leaf_Blight": "Apply fungicides such as mancozeb or chlorothalonil. Ensure proper crop rotation.",
    "Corn_(maize)___healthy": "Your corn plant is healthy. No treatment needed.",
    
    # Grape
    "Grape___Black_rot": "Apply fungicides with myclobutanil or captan. Remove and destroy infected leaves and fruit. Improve air circulation by pruning.",
    "Grape___Esca_(Black_Measles)": "Apply fungicides such as thiophanate-methyl. Prune out infected canes. Ensure proper watering and avoid vine stress.",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Use fungicides containing copper or mancozeb. Remove infected leaves and maintain proper vineyard sanitation.",
    "Grape___healthy": "Your grapevine is healthy. No treatment needed.",
    
    # Orange
    "Orange___Haunglongbing_(Citrus_greening)": "No cure is currently available. Remove and destroy infected trees. Use insecticides to control psyllid vectors.",
    
    # Peach
    "Peach___Bacterial_spot": "Use copper-based bactericides. Avoid overhead watering and prune infected branches.",
    "Peach___healthy": "Your peach tree is healthy. No treatment needed.",
    
    # Pepper
    "Pepper,_bell___Bacterial_spot": "Apply copper-based bactericides. Rotate crops and avoid overhead irrigation.",
    "Pepper,_bell___healthy": "Your pepper plant is healthy. No treatment needed.",
    
    # Potato
    "Potato___Early_blight": "Apply fungicides containing chlorothalonil or mancozeb. Ensure proper crop rotation and avoid excessive irrigation.",
    "Potato___Late_blight": "Use fungicides with chlorothalonil or mancozeb. Remove infected plants and avoid planting in the same soil.",
    "Potato___healthy": "Your potato plant is healthy. No treatment needed.",
    
    # Raspberry
    "Raspberry___healthy": "Your raspberry plant is healthy. No treatment needed.",
    
    # Soybean
    "Soybean___healthy": "Your soybean plant is healthy. No treatment needed.",
    
    # Squash
    "Squash___Powdery_mildew": "Apply sulfur-based fungicides or neem oil. Ensure good air circulation around the plant.",
    
    # Strawberry
    "Strawberry___Leaf_scorch": "Use fungicides like captan or myclobutanil. Remove infected leaves and improve air circulation.",
    "Strawberry___healthy": "Your strawberry plant is healthy. No treatment needed.",
    
    # Tomato
    "Tomato___Bacterial_spot": "Apply copper-based bactericides. Avoid overhead irrigation and ensure proper crop rotation.",
    "Tomato___Early_blight": "Use fungicides containing chlorothalonil or mancozeb. Remove infected leaves and improve air circulation.",
    "Tomato___Late_blight": "Apply fungicides with chlorothalonil or mancozeb. Remove infected plants immediately to prevent spread.",
    "Tomato___Leaf_Mold": "Use fungicides containing copper or sulfur. Increase air circulation and avoid overhead watering.",
    "Tomato___Septoria_leaf_spot": "Apply fungicides like chlorothalonil or mancozeb. Remove infected leaves and avoid water splashing on leaves.",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Use insecticidal soap or neem oil to control mite populations. Ensure proper watering to avoid plant stress.",
    "Tomato___Target_Spot": "Apply fungicides such as copper or mancozeb. Remove infected leaves and increase air circulation.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "No cure is available. Control whitefly populations that spread the virus using insecticides.",
    "Tomato___Tomato_mosaic_virus": "Remove and destroy infected plants. Sterilize tools and avoid tobacco contact with plants.",
    "Tomato___healthy": "Your tomato plant is healthy. No treatment needed."
}

# Function to get treatment based on disease prediction
def get_treatment_recommendation(prediction):
    return treatment_recommendations.get(prediction, "No specific treatment available. Consult an expert.")

# Disease Lifecycle Tracker
if 'disease_history' not in st.session_state:
    st.session_state['disease_history'] = []

def track_disease_lifecycle(image, prediction, severity):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state['disease_history'].append({
        "image": image,
        "prediction": prediction,
        "severity": severity,
        "timestamp": timestamp
    })

def show_disease_lifecycle():
    st.header("Disease Lifecycle Tracker")
    if not st.session_state['disease_history']:
        st.write("No history yet. Upload images to track disease progression.")
    else:
        for entry in st.session_state['disease_history']:
            st.image(entry['image'], caption=f"{entry['timestamp']}: {entry['prediction']} ({entry['severity']})", use_column_width=True)

# Translate text using Google Translate
def translate_text(text, dest_language='en'):
    try:
        translation = translator.translate(text, dest=dest_language)
        return translation.text
    except Exception as e:
        return f"Translation failed: {str(e)}"

# Download model from Google Drive if not found
def download_model_from_drive():
    model_path = "trained_plant_disease_model.keras"
    if not os.path.exists(model_path):
        file_id = "1xtMHfd0fmgpKGZyPr_lNRBZpjA0bl_bF"  # <-- Replace with your real Drive file ID
        url = f"https://drive.google.com/uc?id=1xtMHfd0fmgpKGZyPr_lNRBZpjA0bl_bF"
        st.info("Downloading model from Google Drive. Please wait...")
        gdown.download(url, model_path, quiet=False)

# Predict disease from image
def model_prediction(test_image):
    download_model_from_drive()

    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)
    class_index = np.argmax(predictions)

    disease_labels = [ 'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
        'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
        'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
        'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
        'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
        'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
        'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
        'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
        'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
        'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
        'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
        'Tomato___healthy'
    ]

    return disease_labels[class_index]

# Streamlit Sidebar and App Routing
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition", "Lifecycle Tracker"])

if app_mode == "Home":
    st.header("🌿 PLANT DISEASE RECOGNITION SYSTEM")
    st.image("home_page.jpeg", use_column_width=True)
    st.markdown("""
    Welcome to the **Plant Disease Recognition System**!  
    Upload a leaf image 🌱 and let AI detect the disease, suggest treatments, and support you in multiple languages.
    """)

elif app_mode == "Disease Recognition":
    st.header("🧠 Disease Recognition")

    uploaded_file = st.file_uploader("Upload a plant leaf image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        st.image(uploaded_file, use_column_width=True)

        predicted_disease = model_prediction(uploaded_file)

        selected_language = st.selectbox("🌐 Select Language", ['en', 'hi', 'ml', 'ta', 'te', 'kn', 'gu', 'mr', 'bn', 'pa', 'es', 'fr', 'zh'])

        st.write(f"Predicted Disease: **{predicted_disease}**")
        st.write(f"Translated: **{translate_text(predicted_disease, selected_language)}**")

        treatment = get_treatment_recommendation(predicted_disease)
        st.write(f"Recommended Treatment: {treatment}")
        st.write(f"Translated Treatment: {translate_text(treatment, selected_language)}")

        severity_level = "Moderate"
        track_disease_lifecycle(uploaded_file, predicted_disease, severity_level)

elif app_mode == "Lifecycle Tracker":
    show_disease_lifecycle()
