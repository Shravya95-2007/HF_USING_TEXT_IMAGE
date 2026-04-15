import streamlit as st
import requests

st.title("🎨 Text-to-Image Generator (API-based)")
st.write("Generate images from text prompts using Hugging Face Inference API")

prompt = st.text_area("Enter your image prompt:", height=100)

if st.button("Generate Image"):
    if prompt.strip():
        with st.spinner("Calling API... ⏳"):
            response = requests.post(
                "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5",
                headers={"Authorization": f"Bearer YOUR_HF_API_TOKEN"},
                json={"inputs": prompt}
            )
            if response.status_code == 200:
                st.image(response.content, caption="Generated from your prompt")
            else:
                st.error(f"API error: {response.text}")
    else:
        st.warning("⚠️ Please enter a text prompt.")
