import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av
import fastai
from fastai.vision.all import *
from fastai.learner import load_learner
from PIL import Image
import os
import time
import csv
import requests
import io

# Constants
FEEDBACK_DIR = 'feedback_images'
CSV_FILENAME = os.path.join(FEEDBACK_DIR, 'feedback.csv')
MODEL_URL = "https://huggingface.co/VasuSingh/fastai_waste_classification/resolve/main/model_fastai.pkl?download=true"
#MODEL_PATH = 'model_fastai.pkl'

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

def setup_directories():
    if not os.path.exists(FEEDBACK_DIR):
        os.makedirs(FEEDBACK_DIR)
    if not os.path.exists(CSV_FILENAME):
        with open(CSV_FILENAME, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Image ID', 'Predicted Label', 'Predicted Probability', 'Feedback'])

def download_model(url):
    response = requests.get(url)
    if response.status_code == 200:
        # Debug: Print the first 100 characters of the response content
        print(response.content[:100])

        # Load the model directly from the in-memory bytes buffer
        bytes_buffer = io.BytesIO(response.content)
        try:
            learn = load_learner(bytes_buffer)
            return learn
        except Exception as e:
            print(f"Error loading the model: {e}")
            # Additional error handling can be implemented here
            raise
    else:
        raise Exception(f"Failed to download model: {response.status_code}")

@st.cache_resource
def load_model():
    learn = download_model(MODEL_URL)
    return learn

def save_image(image, label, pred_prob=None):
    timestamp = int(time.time())
    filename = f"{label.replace(' ', '_')}_{timestamp}.png"
    filepath = os.path.join(FEEDBACK_DIR, filename)
    image.save(filepath)
    
    new_entry = [filename, label, str(pred_prob) if pred_prob else "N/A", 'Feedback']
    with open(CSV_FILENAME, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(new_entry)
    st.write(f"Feedback saved. Image stored as: {filename}")

def main():
    setup_directories()
    learn = load_model()
    
    st.title("BeClean")
    st.subheader("Play. Sort. Reward. Repeat.")

    categories = ['battery', 'biological', 'brown-glass', 'cardboard', 'clothes',
                  'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass']

    tab1, tab2 = st.tabs(["Camera", "Result"])

    with tab1:
        webrtc_ctx = webrtc_streamer(key="example",
                                     video_transformer_factory=VideoTransformer,
                                     rtc_configuration=RTC_CONFIGURATION,
                                     media_stream_constraints={"video": True, "audio": False})
        capture_button = st.button("Capture")

    if capture_button and webrtc_ctx.state.playing and webrtc_ctx.video_transformer.frame is not None:
        image = PILImage(webrtc_ctx.video_transformer.frame)
        pred, pred_idx, probs = learn.predict(image)
        prediction_text = f'Prediction: {pred}; Probability: {probs[pred_idx].item() * 100:.2f}%'
        st.session_state.update({'captured_image': webrtc_ctx.video_transformer.frame, 'prediction': prediction_text,
                                 'pred': pred, 'pred_prob': probs[pred_idx].item()})

    with tab2:
        if 'captured_image' in st.session_state:
            st.image(st.session_state['captured_image'], caption=st.session_state['prediction'] if 'prediction' in st.session_state else "No prediction made.")
            feedback = st.radio("Is the prediction correct?", ('Correct', 'Incorrect'), index=1)
            correct_label = st.selectbox("Select the correct label:", categories) if feedback == 'Incorrect' else st.session_state['pred']
            if st.button('Submit Feedback'):
                save_image(st.session_state['captured_image'], correct_label, st.session_state['pred_prob'])

if __name__ == "__main__":
    main()
