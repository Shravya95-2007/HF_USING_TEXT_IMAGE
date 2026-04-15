import streamlit as st
import requests
import os
from io import BytesIO
from PIL import Image

API_URL = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
API_TOKEN = os.getenv("HF_TOKEN")  # read from environment variable

headers = {"Authorization": f"Bearer {API_TOKEN}"}

st.title("🎨 AI Text-to-Image Generator")
prompt = st.text_area("Enter your image prompt:", height=200)

if st.button("Generate Image"):
    if prompt.strip():
        with st.spinner("Calling Hugging Face API... ⏳"):
            response = requests.post(API_URL, headers=headers, json={"inputs": prompt})

            if response.status_code == 200:
                image = Image.open(BytesIO(response.content))
                st.image(image, caption="Generated from your prompt")
            else:
                st.error(f"API error: {response.text}")
    else:
        st.warning("⚠️ Please enter a text prompt.")
