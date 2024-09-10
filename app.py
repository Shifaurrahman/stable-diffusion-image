import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import io

# Load the model once at the beginning (for efficiency)
@st.cache_resource
def load_model():
    model_id = "CompVis/stable-diffusion-v1-4"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionPipeline.from_pretrained(model_id).to(device)
    return pipe

# Generate image function
def generate_image(prompt):
    pipe = load_model()
    image = pipe(prompt).images[0]
    return image

# Streamlit app
def main():
    st.title("Stable Diffusion Image Generator")
    
    prompt = st.text_input("Enter a text prompt:")
    
    if st.button("Generate Image"):
        if prompt:
            with st.spinner('Generating...'):
                image = generate_image(prompt)
                st.image(image, caption=f"Generated Image for: {prompt}", use_column_width=True)
                
                # Option to download the image
                buf = io.BytesIO()
                image.save(buf, format="PNG")
                byte_im = buf.getvalue()
                st.download_button(label="Download Image", data=byte_im, file_name="generated_image.png", mime="image/png")
        else:
            st.error("Please enter a prompt!")

if __name__ == "__main__":
    main()
