import streamlit as st
import requests
import base64

st.title("🎨 AI Text-to-Image Generator")
st.write("Enter a text prompt below, and generate an image!")

# Text Input
prompt = st.text_area("Enter your image prompt:", height=200)

# Button
if st.button("Generate Image"):
    if prompt.strip():
        with st.spinner("Generating image... ⏳"):
            response = requests.post(
                "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5",
                headers={"Authorization": f"baidu/ERNIE-Image-Turbo"},
                json={"inputs": prompt}
            )

            if response.status_code == 200:
                result = response.json()
                # Hugging Face returns a list with base64 image(s)
                if isinstance(result, list) and "generated_image" in result[0]:
                    image_base64 = result[0]["generated_image"]
                    image_bytes = base64.b64decode(image_base64)
                    st.subheader("🖼️ Generated Image:")
                    st.image(image_bytes, caption="Generated from your prompt")
                else:
                    st.error("Unexpected API response format.")
            else:
                st.error(f"API error: {response.text}")
    else:
        st.warning("⚠️ Please enter a text prompt.")
