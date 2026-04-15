import streamlit as st
from diffusers import StableDiffusionPipeline
import torch

# Load the text-to-image model
@st.cache_resource
def load_text2image():
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")  # Use GPU if available
    return pipe

text2image = load_text2image()

# Streamlit UI
st.title("🎨 AI Text-to-Image Generator")
st.write("Enter a text prompt below, and generate an image!")

# Text Input
prompt = st.text_area("Enter your image prompt:", height=100)

# Image Parameters
guidance_scale = st.slider("Guidance Scale", min_value=1.0, max_value=15.0, value=7.5)
num_inference_steps = st.slider("Inference Steps", min_value=10, max_value=100, value=50)

if st.button("Generate Image"):
    if prompt.strip():
        with st.spinner("Creating image... ⏳"):
            image = text2image(prompt, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps).images[0]
            st.subheader("🖼️ Generated Image:")
            st.image(image, caption="Generated from your prompt")
    else:
        st.warning("⚠️ Please enter a text prompt to generate an image.")
