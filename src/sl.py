import logging
from utils import setup_logging
import requests
import streamlit as st

def run():
    LOG_CONFIG = "../conf/base/logging.yaml"

    logger = logging.getLogger(__name__)
    logger.info("Setting up logging configuration.")
    setup_logging(LOG_CONFIG)

    logger.info("Initialising Front-end...")

    st.title("Pneumonia Diagnosis for X-Ray Images")

    uploaded_file = st.file_uploader(
        label="Upload a X-ray image", type=["jpg", "jpeg", "png"]
    )

    if st.button("Predict"):
        logger.info("Sending Request for Prediction")
        files = {"file": uploaded_file.getvalue()}
        response = requests.post("http://localhost:8000/predict",files=files)
        st.write(response.json())
    return

if __name__ == "__main__":
    run()
