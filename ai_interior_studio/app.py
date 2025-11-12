import streamlit as st
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline

# ---------------------------
# Load Models (cached)
# ---------------------------
@st.cache_resource
def load_text2img_model():
    model_id = "SG161222/Realistic_Vision_V5.1_noVAE"  # highly realistic interiors
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16
    )
    if torch.backends.mps.is_available():
        pipe = pipe.to("mps")
    else:
        pipe = pipe.to("cpu")
    return pipe

@st.cache_resource
def load_img2img_model():
    model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16
    )
    if torch.backends.mps.is_available():
        pipe = pipe.to("mps")
    else:
        pipe = pipe.to("cpu")
    return pipe


text2img_pipe = load_text2img_model()
img2img_pipe = load_img2img_model()


# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="AI Interior Studio", layout="wide")
st.title("🏠 AI Interior Design Studio")

tabs = st.tabs(["✨ Text-to-Image", "🖼️ Image-to-Image", "🎨 Room Dimensions"])

# ==========================
# TAB 1: TEXT → IMAGE
# ==========================
with tabs[0]:
    st.subheader("Generate interiors from a text prompt")

    prompt = st.text_area("Enter your interior design idea:", 
                          "A modern minimalistic living room with a white sofa, wooden floor, and large windows")

    color = st.color_picker("Pick a primary room color:", "#ffffff")
    style = st.selectbox("Choose a style:", ["Modern", "Minimalist", "Scandinavian", "Luxury", "Rustic"])

    if st.button("Generate Designs", key="txt2img_btn"):
        with st.spinner("Generating realistic interiors..."):
            final_prompt = f"{prompt}, in {style} style, dominant color {color}, ultra-realistic, photorealistic rendering"
            images = text2img_pipe(
                final_prompt,
                num_inference_steps=25,
                guidance_scale=7.5,
                num_images_per_prompt=2,
                height=512, width=512
            ).images

            st.image(images, caption=["Design Option 1", "Design Option 2"], use_column_width=True)


# ==========================
# TAB 2: IMAGE → IMAGE
# ==========================
with tabs[1]:
    st.subheader("Redesign from an uploaded image")

    uploaded_img = st.file_uploader("Upload your room photo:", type=["jpg", "jpeg", "png"])
    variation_prompt = st.text_input("What do you want to change?", "Change sofa to beige leather, add indoor plants")

    if uploaded_img is not None and st.button("Generate Variations", key="img2img_btn"):
        with st.spinner("Generating new designs..."):
            from PIL import Image
            init_img = Image.open(uploaded_img).convert("RGB").resize((512, 512))

            images = img2img_pipe(
                prompt=variation_prompt + ", ultra-realistic, photorealistic rendering",
                image=init_img,
                strength=0.75,
                guidance_scale=8.0,
                num_images_per_prompt=2
            ).images

            st.image(images, caption=["Redesign Option 1", "Redesign Option 2"], use_column_width=True)


# ==========================
# TAB 3: ROOM DIMENSIONS
# ==========================
with tabs[2]:
    st.subheader("Generate a design just from room dimensions")

    width = st.number_input("Room Width (meters)", 3.0)
    length = st.number_input("Room Length (meters)", 4.0)
    height = st.number_input("Room Height (meters)", 2.5)

    furniture = st.selectbox("Furniture Type:", ["Living Room", "Bedroom", "Office", "Kitchen"])

    if st.button("Generate Layout", key="dim_btn"):
        with st.spinner("Generating layout..."):
            dim_prompt = f"A {furniture} interior, room size {width}x{length}x{height} meters, ultra-realistic design, photorealistic rendering"
            images = text2img_pipe(
                dim_prompt,
                num_inference_steps=25,
                guidance_scale=7.5,
                num_images_per_prompt=2,
                height=512, width=512
            ).images

            st.image(images, caption=["Layout Option 1", "Layout Option 2"], use_column_width=True)

            st.markdown("💡 Furniture alignment suggestion: Place larger furniture along the longest wall, keep walking paths at least 0.9m wide, and use lighter colors to make the room appear bigger.")


st.markdown("---")
st.info("✅ Pro Tip: Use higher resolution (768x768) for better realism, but keep in mind it may take longer on Mac M2.")
