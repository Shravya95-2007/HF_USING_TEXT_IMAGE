import streamlit as st
from diffusers import StableDiffusionPipeline
import torch

# Load model (uses GPU) - This should ideally be cached or loaded once
@st.cache_resource
def load_model():
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")
    return pipe

pipe = load_model()

st.title("Stable Diffusion Image Generator")

prompt = st.text_input("Enter your image prompt:", "a photograph of an astronaut riding a horse")

if st.button("Generate Image"):
    if prompt:
        with st.spinner("Generating image..."):
            # Generate image
            image = pipe(prompt, num_inference_steps=20).images[0]
            st.image(image, caption=prompt)
    else:
        st.warning("Please enter a prompt.")
