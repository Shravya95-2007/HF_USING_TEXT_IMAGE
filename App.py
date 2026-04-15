import streamlit as st
import requests
from io import BytesIO
from PIL import Image

# Hugging Face API endpoint and token
API_URL = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
API_TOKEN = "HF_TOKEN"   # replace with your token

headers = {"Authorization": f"Bearer {API_TOKEN}"}

st.title("🎨 AI Text-to-Image Generator")
st.write("Enter a text prompt below, and generate an image!")

# Text Input
prompt = st.text_area("Enter your image prompt:", height=200)

# Generate Button
if st.button("Generate Image"):
    if prompt.strip():
        with st.spinner("Generating image... ⏳"):
            response = requests.post(API_URL, headers=headers, json={"inputs": prompt})

            if response.status_code == 200:
                # Hugging Face returns raw image bytes for diffusion models
                image = Image.open(BytesIO(response.content))
                st.subheader("🖼️ Generated Image:")
                st.image(image, caption="Generated from your prompt")
            else:
                st.error(f"API error: {response.text}")
    else:
        st.warning("⚠️ Please enter a text prompt.")
