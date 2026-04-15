import streamlit as st
from transformers import pipeline
from PIL import Image

# Load the text-to-image pipeline
@st.cache_resource
def load_text2image():
    return pipeline("text-to-image", model="baidu/ERNIE-Image-Turbo")

generator = load_text2image()

# Streamlit UI
st.title("🎨 AI Text-to-Image Generator")
st.write("Enter a text prompt below, and generate an image!")

# Text Input
prompt = st.text_area("Enter your image prompt:", height=200)

# Image Parameters
num_images = st.slider("Number of Images", min_value=1, max_value=3, value=1)
guidance_scale = st.slider("Guidance Scale", min_value=1.0, max_value=15.0, value=7.5)

# Generate Button
if st.button("Generate Image"):
    if prompt.strip():
        with st.spinner("Generating image... ⏳"):
            images = generator(
                prompt,
                num_images_per_prompt=num_images,
                guidance_scale=guidance_scale
            )
            st.subheader("🖼️ Generated Image(s):")
            for i, img in enumerate(images):
                if isinstance(img, Image.Image):
                    st.image(img, caption=f"Image {i+1}")
                else:
                    st.error("Unexpected output format from pipeline.")
    else:
        st.warning("⚠️ Please enter a text prompt.")
