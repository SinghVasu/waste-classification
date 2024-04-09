# -*- coding: utf-8 -*-

# Created on Thu Mar 14 03:25:14 2024

# @author: vsingh1

import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
from fastai.vision.all import load_learner, PILImage
import av

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

@st.cache_resource
def load_model():
    model_path = 'model_fastai.pkl'  # Path to your FastAI exported model file
    try:
        learn = load_learner(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None
    return learn

learn = load_model()

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.frame = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        self.frame = frame
        return frame

    def get_frame(self):
        return self.frame

def main():
    st.title("Real-time Waste Classification App")
    result_placeholder = st.empty()

    # Create tabs for the camera and the results
    tab1, tab2 = st.tabs(["Camera", "Results"])

    with tab1:
        webrtc_ctx = webrtc_streamer(key="example",
                                     video_transformer_factory=VideoTransformer,
                                     rtc_configuration=RTC_CONFIGURATION,
                                     media_stream_constraints={"video": True, "audio": False})

        if st.button("Capture") and learn is not None:
            if webrtc_ctx.state.playing and webrtc_ctx.video_transformer:
                frame = webrtc_ctx.video_transformer.get_frame()
                if frame is not None:
                    captured_image = frame.to_image().convert('RGB')

                    # Display the captured image in the results tab
                    with tab2:
                        st.image(captured_image, caption='Captured Image', use_column_width=True)

                        # Convert PIL image to FastAI's Image object and predict
                        img = PILImage.create(captured_image)
                        pred_class, pred_idx, outputs = learn.predict(img)

                        # Display the prediction result
                        st.write(f"Prediction: {learn.dls.vocab[pred_class]}")

if __name__ == "__main__":
    main()
