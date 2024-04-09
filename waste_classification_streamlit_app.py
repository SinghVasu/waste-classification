# -*- coding: utf-8 -*-

# Created on Thu Mar 14 03:25:14 2024

# @author: vsingh1


import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av
import numpy as np
import tensorflow as tf


RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# Load and compile the model explicitly
@st.cache_resource
def load_model():
    model_path = 'waste_classifier_model_tf.h5'
    model = tf.keras.models.load_model(model_path, compile=False)
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

model = load_model()

# Define the labels
labels = [
    'battery', 'biological', 'brown-glass', 'cardboard', 'clothes',
    'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass'
]

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.frame = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        self.frame = frame  # Store the current frame
        return frame  # Return the original frame

    def get_frame(self):
        return self.frame

def main():
    st.title("BeClean")
    result_placeholder = st.empty()

    # Create tabs for the camera and the results
    tab1, tab2 = st.tabs(["Camera", "Results"])

    with tab1:
        webrtc_ctx = webrtc_streamer(key="example", 
                                     video_transformer_factory=VideoTransformer,
                                     rtc_configuration=RTC_CONFIGURATION, 
                                     media_stream_constraints={"video": True, "audio": False})

        if st.button("Capture"):
            if webrtc_ctx.state.playing and webrtc_ctx.video_transformer:
                # Access the frame from the video transformer
                frame = webrtc_ctx.video_transformer.get_frame()
                if frame is not None:
                    captured_image = frame.to_image().convert('RGB')

                    # Display the captured image in the results tab
                    with tab2:
                        st.image(captured_image, caption='Captured Image', use_column_width=True)

                        # Process the image for classification
                        img = captured_image.resize((180, 180))
                        img_array = np.expand_dims(np.array(img), axis=0)
                        predictions = model.predict(img_array)
                        predicted_class = labels[np.argmax(predictions[0])]
                        #confidence = np.max(predictions[0]) * 100

                        # Display the prediction result
                        st.write(f"Prediction: {predicted_class}")

if __name__ == "__main__":
    main()
